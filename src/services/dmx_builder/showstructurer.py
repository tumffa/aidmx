import math
import random
import array
from scipy import interpolate

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe_size = data_manager.universe.get("size", 512)
        self.universe = {k: v for k, v in data_manager.universe.items() if k != "size"}

        # Dimmer map for help with seperating dimmer commands
        # and fixture address map
        self.fixture_addresses = {}
        self.fixture_dimmer_map = {}
        self.fixture_id_qlc_id_map = {}
        self.fixture_dimmerrange = {}  # Maps fixture_id to (min, max) dimmer range
        for group_name, group in self.universe.items():
            for fixture_key, fixture in group.items():
                fixture_id = fixture["id"]
                fixture_qlc_id = fixture["qlc_id"]
                self.fixture_id_qlc_id_map[fixture_id] = fixture.get("qlc_id", fixture_id)
                self.fixture_addresses[fixture_qlc_id] = fixture["address"]
                dimmer_channel = fixture["dimmer"]
                self.fixture_dimmer_map[fixture_id] = dimmer_channel
                # Store dimmerrange (default [0, 255] if not specified)
                dimmerrange = fixture.get("dimmerrange", [0, 255])
                if isinstance(dimmerrange, (list, tuple)) and len(dimmerrange) >= 2:
                    self.fixture_dimmerrange[fixture_id] = (int(dimmerrange[0]), int(dimmerrange[1]))
                else:
                    self.fixture_dimmerrange[fixture_id] = (0, 255)

        self.dimmer_update_fq = 33 # ms

    def get_songdata(self, name):
        struct = self.dm.get_struct_data(name)
        song_data = self.dm.get_song(name)
        return struct, song_data
    
    def create_show(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        return show
    
    def _light_strength_envelope_function(self, light_strength_envelope):
        """
        Converts a discrete light strength envelope into a callable function.
        This allows calculating envelope values at any time without storing large arrays.
        
        Args:
            light_strength_envelope: Dictionary with 'times' and 'values' lists
            
        Returns:
            A function that takes a time value and returns the envelope strength at that time
        """
        if not light_strength_envelope:
            return lambda t: 1.0
            
        env_times = light_strength_envelope.get("times", [])
        env_values = light_strength_envelope.get("values", [])
        
        if not env_times or not env_values:
            print("Warning: Empty envelope times or values.")
            return lambda t: 1.0
        
        # Create interpolation function (linear interpolation)
        interp_func = interpolate.interp1d(
            env_times, 
            env_values,
            bounds_error=False,     # Don't raise error for out-of-bounds
            fill_value=(0.01, 0.01)   # Baseline value for out-of-bounds
        )
        
        # Return a closure function that handles the interpolation
        def envelope_function(t):
            return float(interp_func(t))
            
        return envelope_function

    def _setfixture(self, fixture, channel, value, comment=None, scale_dimmer=None):
        value = max(0, min(255, value))
        return (fixture, channel, value, scale_dimmer)
    
    def map_to_qlc_id(self, cmd):
        fixture_id, channel, value, scale_dimmer = cmd
        qlc_id = self.fixture_id_qlc_id_map.get(fixture_id)
        if qlc_id is None:
            raise ValueError(f"Fixture ID {fixture_id} not found in QLC mapping")
        return (qlc_id, channel, value, scale_dimmer)
        
    def _qlc_setfixture(self, fixture, channel, value, comment=""):
        return f"setfixture:{fixture} ch:{channel} val:{value} //{comment}"

    def _wait(self, time, comment=""):
        time = max(0, int(round(float(time))))
        return int(time)
    
    def _qlc_wait(self, time, comment=""):
        time = max(0, int(round(float(time))))
        return f"wait:{int(time)} //{comment}"
    
    def _execute(self, command, comment=""):
        return f"systemcommand:{command} //{comment}"
    
    def combine(self, queues, length=None, strobe_ranges=None):
        """
        Combines multiple chaser-compiled scripts (plain lists) into a single sequence per time segment.

        Notes on timebases:
        - current_time_ms is segment-local (0..segment_length). Segment-start waits are added later.
        - strobe_ranges are segment-local milliseconds (0..segment_length).
        """

        queue_priorities = []
        for q in queues:
            scripts = q.get("script")
            if isinstance(scripts, dict) and "queue" in scripts:
                scripts = scripts["queue"].get_queue() if hasattr(scripts["queue"], "get_queue") else scripts["queue"]
            elif hasattr(scripts, "get_queue"):
                scripts = scripts.get_queue()
            if not isinstance(scripts, list):
                scripts = [] if scripts is None else list(scripts) if isinstance(scripts, (tuple, list)) else []
            q["script"] = scripts
            if not scripts:
                queue_priorities.append(None)
            elif isinstance(scripts[0], (int, float)):
                queue_priorities.append(max(0, int(round(float(scripts[0])))))
            else:
                queue_priorities.append(0)

        fixture_states = {}
        segment = []
        segment_dimmers = []
        current_time_ms = 0

        while current_time_ms < length and any(q["script"] for q in queues):
            valid_indices = [i for i, q in enumerate(queues)
                                if queues[i].get("script") and queue_priorities[i] is not None]
            if not valid_indices:
                break

            # Compute strobe status at current segment-local time
            in_strobe_now = False
            if strobe_ranges:
                try:
                    in_strobe_now, strobe_end = self.check_strobe_ranges(current_time_ms, strobe_ranges)
                except Exception:
                    in_strobe_now = False

            min_priority = min(queue_priorities[i] for i in valid_indices)
            script_idx = next(i for i in valid_indices if queue_priorities[i] == min_priority)

            scripts = queues[script_idx]["script"]
            if not scripts:
                queue_priorities[script_idx] = None
                continue

            # Flatten head lists IN-PLACE
            while scripts and isinstance(scripts[0], list):
                scripts[0:1] = scripts[0]

            exiting_strobe_range = False

            # if entry is wait
            if isinstance(scripts[0], (int, float)):
                wait_ms = max(0, int(round(float(scripts.pop(0)))))
                if in_strobe_now and strobe_end is not None:
                    if current_time_ms + wait_ms >= strobe_end:
                        wait_ms = strobe_end - current_time_ms
                        exiting_strobe_range = True
                        
                current_time_ms += wait_ms

                for i, q in enumerate(queues):
                    if i == script_idx:
                        continue
                    other_scripts = q["script"]
                    if other_scripts and isinstance(other_scripts[0], (int, float)):
                        other_scripts[0] = max(0, int(round(float(other_scripts[0]))) - wait_ms)
                    queue_priorities[i] = max(0, queue_priorities[i] - wait_ms)

                segment.append(self._wait(wait_ms))
                segment_dimmers.append(self._wait(wait_ms))

            # if entry is command
            elif isinstance(scripts[0], (list, tuple)):
                cmds = []
                dimmer_cmds = []

                while scripts and not isinstance(scripts[0], (int, float)):
                    cmd = scripts.pop(0)
                    fix_id, channel, value, scale = cmd
                    if self.fixture_dimmer_map[fix_id] == channel:
                        dimmer_cmds.append(self.map_to_qlc_id(cmd))
                    else:
                        cmds.append(self.map_to_qlc_id(cmd))
                    if fix_id not in fixture_states:
                        fixture_states[fix_id] = {}
                    fixture_states[fix_id][channel] = (value, scale)

                queue_priorities[script_idx] = scripts[0] if (scripts and isinstance(scripts[0], (int, float))) else float('inf')

                # Gate emission strictly by current strobe status
                if not in_strobe_now:
                    if cmds:
                        segment.append(cmds)
                    if dimmer_cmds:
                        segment_dimmers.append(dimmer_cmds)

            # if we exited a strobe range, restore fixture states
            if exiting_strobe_range:
                cmds = []
                dimmer_cmds = []
                for fix_id, channels in fixture_states.items():
                    for channel, (value, scale) in channels.items():
                        cmd = (fix_id, channel, value, scale)
                        if self.fixture_dimmer_map[fix_id] == channel:
                            dimmer_cmds.append(self.map_to_qlc_id(cmd))
                        else:
                            cmds.append(cmd)
                if cmds:
                    segment.append(cmds)
                if dimmer_cmds:
                    segment_dimmers.append(dimmer_cmds)

        return segment, segment_dimmers
    
    def segment_strobe_ranges(self, segment_start, segment_end, strobe_ranges):
        """
        Clip song-absolute strobe ranges in milliseconds to [segment_start_ms, segment_end_ms],
        convert to segment-local milliseconds, and normalize (drop sub-1ms windows, sort, merge overlaps/adjacents).
        Returns list of (start_ms, end_ms) in segment-local milliseconds.
        """
        if not strobe_ranges:
            return []

        # Segment boundaries in ms
        try:
            seg_start = int(round(float(segment_start)))
            seg_end = int(round(float(segment_end)))
        except Exception:
            return []
        if seg_end <= seg_start:
            return []

        # Clip to segment and convert to local milliseconds
        clipped = []
        for r in strobe_ranges:
            if not isinstance(r, (tuple, list)) or len(r) < 2:
                continue
            try:
                s_abs = int(round(float(r[0]))); e_abs = int(round(float(r[1])))
            except Exception:
                continue
            if e_abs <= s_abs:
                continue
            # overlap?
            if e_abs <= seg_start or s_abs >= seg_end:
                continue
            cs = max(s_abs, seg_start)
            ce = min(e_abs, seg_end)
            if ce <= cs:
                continue
            # convert to segment-local milliseconds
            start_rel_ms = cs - seg_start
            end_rel_ms = ce - seg_start
            clipped.append((start_rel_ms, end_rel_ms))

        if not clipped:
            return []

        # Drop near-zero ranges (<1 ms), sort and merge
        EPS_MS = 1  # 1 ms
        clipped = [(s, e) for (s, e) in clipped if (e - s) >= EPS_MS]
        if not clipped:
            return []

        clipped.sort(key=lambda x: x[0])

        merged = []
        cs, ce = clipped[0]
        for s, e in clipped[1:]:
            # merge overlaps or adjacency within EPS_MS
            if s <= ce + EPS_MS:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))

        return merged

    def check_strobe_ranges(self, current_time_ms, strobe_ranges):
        """
        Returns (in_strobe_range, current_range_end_ms) for segment-local ms.
        strobe_ranges: [(start_ms, end_ms), ...] in milliseconds relative to the segment start.
        """
        t = int(round(float(current_time_ms)))
        if not strobe_ranges:
            return False, None
        for r in strobe_ranges:
            try:
                s = int(round(float(r[0]))); e = int(round(float(r[1])))
            except Exception:
                continue
            if s <= t < e:
                return True, e
        return False, None

    def restore_fixture_states(self, fixture_dimmers, segment, segment_dimmers):
        """
        Restores the fixture dimmer states to their original values. This is done when exiting a strobe range.
        
        Args:
            fixture_dimmers: Dictionary of current fixture dimmer values
            segment_dimmers: Dimmer command segment

        Returns:
            Tuple of (segment, segment_dimmers) with commands for restored dimmer values
        """
        for fixture_id, info in fixture_dimmers.items():
            for channel, value in info.items():
                if channel == self.fixture_dimmer_map.get(fixture_id, None):
                    # Ensure restored dimmers keep scaling enabled
                    dimmer_command = self._setfixture(
                        fixture_id, channel, value,
                        scale_dimmer="both"
                    )
                    segment_dimmers.append(dimmer_command)
                else:
                    cmd = self._setfixture(fixture_id, channel, value)
                    segment.append(cmd)
        return segment, segment_dimmers

    def generate_show(self, name, qxw, strobes, simple, qlc_delay, qlc_lag):
        """
            Generates all functions, buttons, etc. for a QLC+ file. 
        Args:
            name (str): Name of the show to generate
            qxw (Object): object that handles qlc+ file generation 
            strobes (bool, optional): Whether or not to include strobe effects. Defaults to False.
            simple (bool, optional): Whether to use simple mode (only a simple color chaser). Defaults to False.
            qlc_delay (int, optional): Delay in ms to add to each QLC+ script
            qlc_lag (int, optional): Used to scale wait times down to account for lag
        """
        print(f"----Combining scripts for show")
        ola_scripts = []  # OLA scripts
        qlc_scripts = []  # QLC+ scripts
        function_names = [] # list that holds names of beforementioned scripts
        show = self.create_show(name) # holds information about song
        segments = show.struct["segments"] # segments of the song
    
        def _construct_strobe(universe, chaser_name, length):
            from src.services.dmx_builder.chaser import strobe
            result = {}
            script = strobe(universe=universe, length=length)
            result["name"] = chaser_name
            # Return plain script; start_time will be added by combine
            result["script"] = script
            return result

        # Add strobe scripts
        onset_parts = None
        if strobes:
            onset_parts = show.struct["onset_parts"]
            for part in onset_parts:
                start = part[0]
                end = part[1]
                length = (end - start)

                strobe = _construct_strobe(
                    universe=self.universe,
                    chaser_name=f"strobechaser{part[0]}",
                    length=(end - start)
                )

                scripts, _ = self.combine(
                    [strobe],
                    length=length,
                    strobe_ranges=None
                )
                scripts.insert(0, self._wait(start))
                ola_scripts.append(scripts)

                qlc_scripts.append(self.convert_scripts_to_qlc_format([scripts], qlc_delay=qlc_delay, qlc_lag=qlc_lag, is_dimmer=False)[0])
                function_names.append(f"strobe{part[0]}-{part[1]}")

        # Add scripts for each segment in the song
        i = 0
        if segments[0]["label"] == "start":
            i += 1
        onefocus = (len(show.struct["focus"]) == 1) # check how many segments are energetic i.e. verse/chorus/inst

        def _construct_queue_chaser(universe, chaser_name, length, interval):
            from src.services.dmx_builder.chaser import color_pulse
            result = {}
            script = color_pulse(universe=universe, interval=interval, length=length)
            result["name"] = chaser_name
            # Return plain script; start_time will be added by combine
            result["script"] = script
            return result

        for i in range(i, len(segments)):
            start_time_ms = segments[i]["start"]*1000
            end_time_ms = segments[i]["end"]*1000

            length = (end_time_ms - start_time_ms)
            queues = []

            queues.append(_construct_queue_chaser(
                universe=self.universe,
                chaser_name=f"mainchaser{i}",
                length=length,
                interval=show.beatinterval*1000
                )
            )

            strobe_ranges = onset_parts
            segment_strobe_ranges = self.segment_strobe_ranges(
                segment_start=start_time_ms,
                segment_end=end_time_ms,
                strobe_ranges=strobe_ranges
            ) if strobe_ranges else None

            segment_queue, segment_dimmers = self.combine(
                queues,
                length=length,
                strobe_ranges=segment_strobe_ranges,
            )
            light_strength_envelope = segments[i]["drum_analysis"]["light_strength_envelope"]
            segment_dimmers = self.scale_dimmer_with_envelope(start_time_ms, segment_dimmers, light_strength_envelope, strobe_ranges=strobe_ranges)

            # Add start time wait to script first index
            if start_time_ms > 0:
                segment_queue.insert(0, self._wait(start_time_ms))
                segment_dimmers.insert(0, self._wait(start_time_ms))

            # OLA scripts
            ola_scripts.append(segment_queue)
            ola_scripts.append(segment_dimmers)

            # QLC+ scripts (queue and dimmer, lag scaling only on dimmer)
            qlc_scripts.append(self.convert_scripts_to_qlc_format([segment_queue], qlc_delay=qlc_delay, qlc_lag=qlc_lag, is_dimmer=False)[0])
            qlc_scripts.append(self.convert_scripts_to_qlc_format([segment_dimmers], qlc_delay=qlc_delay, qlc_lag=qlc_lag, is_dimmer=True)[0])

            function_names.append(str(segments[i]["start"]))
            function_names.append(str(segments[i]["start"]) + "_dimmers")

        ola_result = self.combine_scripts_to_ola_format(ola_scripts)

        result = {
            "ola": ola_result,
            "qlc": {
                "scripts": qlc_scripts,
                "function_names": function_names
            }
        }
        return result
    
    def scale_dimmer_with_envelope(self, start_ms, segment_dimmers, light_strength_envelope, strobe_ranges=None):
        """
        Per-command envelope scaling with synthetic updates during waits.
        Flags: both → beat_flow_ranges (source 0=beat,1=flow,2=snare), beat, flow, snare. None → no scaling.
        """
        if not segment_dimmers:
            return segment_dimmers

        # Helper: suppress ANY emitted dimmer changes during strobes
        def _in_strobe(abs_ms: int) -> bool:
            if not strobe_ranges:
                return False
            try:
                in_range, end = self.check_strobe_ranges(int(abs_ms), strobe_ranges)
                return bool(in_range)
            except Exception:
                return False

        env = light_strength_envelope or {}
        beat_env = env.get("beat") or {}
        flow_env = env.get("flow") or {}
        snare_env = env.get("snare") or {}

        beat_fn = self._light_strength_envelope_function(beat_env) if beat_env else (lambda t: 1.0)
        flow_fn = self._light_strength_envelope_function(flow_env) if flow_env else (lambda t: 1.0)
        snare_fn = self._light_strength_envelope_function(snare_env) if snare_env else (lambda t: 1.0)

        def _to_ms_val(v):
            try:
                f = float(v)
            except Exception:
                return None
            return int(round(f)) if f >= 1000.0 else int(round(f * 1000.0))

        def _dict_range_to_ms(d):
            if "start_ms" in d and "end_ms" in d:
                s = int(d["start_ms"]); e = int(d["end_ms"])
            elif "start_s" in d and "end_s" in d:
                s = _to_ms_val(d["start_s"]); e = _to_ms_val(d["end_s"])
            else:
                s_raw = d.get("start_ms", d.get("start_s", d.get("start", d.get("begin", d.get("from")))))
                e_raw = d.get("end_ms", d.get("end_s", d.get("end", d.get("to"))))
                s = _to_ms_val(s_raw) if s_raw is not None else None
                e = _to_ms_val(e_raw) if e_raw is not None else None
            return s, e

        def _coerce_pairs_with_source(ranges_like, default_source=0):
            out = []
            for r in ranges_like:
                try:
                    if isinstance(r, (tuple, list)) and len(r) >= 2:
                        s = _to_ms_val(r[0]); e = _to_ms_val(r[1])
                        src = default_source
                        if len(r) >= 3:
                            try:
                                src = int(r[2])
                            except Exception:
                                src = default_source
                        if s is not None and e is not None and e > s:
                            src_norm = 2 if src == 2 else (1 if src == 1 else 0)
                            out.append({"start_ms": s, "end_ms": e, "source": src_norm})
                    elif isinstance(r, dict):
                        s, e = _dict_range_to_ms(r)
                        if s is not None and e is not None and e > s:
                            src_norm = 2 if default_source == 2 else (1 if default_source == 1 else 0)
                            out.append({"start_ms": s, "end_ms": e, "source": src_norm})
                except Exception:
                    continue
            out.sort(key=lambda x: x["start_ms"])
            return out

        def _coerce_simple_ranges(ranges_like):
            out = []
            for r in ranges_like:
                try:
                    if isinstance(r, (tuple, list)) and len(r) >= 2:
                        s = _to_ms_val(r[0]); e = _to_ms_val(r[1])
                    elif isinstance(r, dict):
                        s, e = _dict_range_to_ms(r)
                    else:
                        continue
                    if s is not None and e is not None and e > s:
                        out.append({"start_ms": s, "end_ms": e})
                except Exception:
                    continue
            out.sort(key=lambda x: x["start_ms"])
            return out

        both_ranges = _coerce_pairs_with_source(env.get("beat_flow_ranges") or [], default_source=0)
        if not both_ranges:
            both_ranges = _coerce_pairs_with_source((beat_env.get("active_ranges") or []), default_source=0)
        beat_ranges = _coerce_simple_ranges(beat_env.get("active_ranges") or [])
        flow_ranges = _coerce_simple_ranges(flow_env.get("active_ranges") or [])
        snare_ranges = _coerce_simple_ranges(snare_env.get("active_ranges") or [])

        def _in_ranges_simple(ranges, t_ms):
            for r in ranges:
                if r["start_ms"] <= t_ms < r["end_ms"]:
                    return True
            return False

        def _source_in_both(t_ms):
            for r in both_ranges:
                if r["start_ms"] <= t_ms < r["end_ms"]:
                    return r.get("source", 0)
            return None

        def _fn_for_flag_at_time(flag, t_ms):
            f = (flag or "").strip().lower()
            if not f:
                return None
            # Always apply the envelope function; it already returns baseline (e.g., 0.01) outside ranges
            if f == "beat":
                return beat_fn
            if f == "flow":
                return flow_fn
            if f == "snare":
                return snare_fn
            if f == "both":
                # Prefer the mapped source if present; otherwise default to beat
                src = _source_in_both(t_ms)
                return flow_fn if src == 1 else snare_fn if src == 2 else beat_fn
            return None

        def _scale_value(value, t_ms, fn, fixture_id=None):
            if fn is None:
                return value
            t_sec = t_ms / 1000.0
            strength = float(fn(t_sec))
            # Scale within dimmerrange bounds: min + (value - min) * strength
            # This ensures envelope=0 gives min (not 0), and envelope=1 gives original value
            min_val, max_val = self.fixture_dimmerrange.get(fixture_id, (0, 255))
            scaled = min_val + (float(value) - min_val) * strength
            return max(0, min(255, int(round(scaled))))

        # Track per-fixture last original value and selected flag
        flagged = {}  # (fixture_id, channel) -> {"base": int, "flag": str}

        scaled = []
        seg_ms = 0          # segment-local time (starts after the first alignment wait)
        abs_ms = start_ms
        update_frequency_ms = max(1, int(getattr(self, "dimmer_update_fq", 33)))
        for entry in segment_dimmers:
            if isinstance(entry, int):
                wait_ms = max(0, int(round(float(entry))))
                if wait_ms == 0:
                    scaled.append(0)
                    continue

                end_ms = seg_ms + wait_ms
                while seg_ms < end_ms:
                    delta = min(update_frequency_ms, end_ms - seg_ms)
                    scaled.append(int(delta))
                    seg_ms += delta
                    abs_ms += delta

                    # Synthetic scaled snapshot for all flagged fixtures active at this time,
                    # BUT NEVER during strobe ranges.
                    if _in_strobe(abs_ms):
                        continue

                    batch = []
                    for (fx, ch), info in flagged.items():
                        fn = _fn_for_flag_at_time(info["flag"], seg_ms)
                        if fn is None:
                            continue
                        val = _scale_value(info["base"], seg_ms, fn, fixture_id=fx)
                        batch.append((int(fx), int(ch), int(val), info["flag"]))
                    if batch:
                        scaled.append(batch)
                continue

            # For non-wait entries, suppress EMISSION during strobes, but still update tracking
            # so the envelope resumes correctly after the strobe.
            if isinstance(entry, tuple) and len(entry) >= 3:
                fixture_id, channel, value = int(entry[0]), int(entry[1]), int(entry[2])
                flag = entry[3] if len(entry) >= 4 else None

                if flag:
                    flagged[(fixture_id, channel)] = {"base": int(value), "flag": str(flag).lower()}
                else:
                    if (fixture_id, channel) in flagged:
                        flagged[(fixture_id, channel)]["base"] = int(value)

                if _in_strobe(abs_ms):
                    continue

                fn = _fn_for_flag_at_time(flag, seg_ms) if flag else None
                new_val = _scale_value(int(value), seg_ms, fn, fixture_id=fixture_id)
                scaled.append((fixture_id, channel, new_val, flag))
                continue

            if isinstance(entry, list):
                # Update tracking for all items, but skip emitting the batch if in strobe
                batch_out = []
                seen = set()

                for sub in entry:
                    if isinstance(sub, tuple) and len(sub) >= 3:
                        fixture_id, channel, value = int(sub[0]), int(sub[1]), int(sub[2])
                        flag = sub[3] if len(sub) >= 4 else None

                        if flag:
                            flagged[(fixture_id, channel)] = {"base": int(value), "flag": str(flag).lower()}
                        else:
                            if (fixture_id, channel) in flagged:
                                flagged[(fixture_id, channel)]["base"] = int(value)

                        if not _in_strobe(abs_ms):
                            fn = _fn_for_flag_at_time(flag, seg_ms) if flag else None
                            new_val = _scale_value(int(value), seg_ms, fn, fixture_id=fixture_id)
                            batch_out.append((fixture_id, channel, new_val, flag))
                            seen.add((fixture_id, channel))
                    else:
                        if not _in_strobe(abs_ms):
                            batch_out.append(sub)

                if _in_strobe(abs_ms):
                    continue

                scaled.append(batch_out)

                # Inject synthetic scaled snapshot for flagged fixtures not in this batch (also gated)
                synth = []
                for (fx, ch), info in flagged.items():
                    if (fx, ch) in seen:
                        continue
                    fn = _fn_for_flag_at_time(info["flag"], seg_ms)
                    if fn is None:
                        continue
                    val = _scale_value(info["base"], seg_ms, fn, fixture_id=fx)
                    synth.append((int(fx), int(ch), int(val), info["flag"]))
                if synth:
                    scaled.append(synth)
                continue

            # Passthrough for unknown types (but suppress during strobes)
            if _in_strobe(abs_ms):
                continue
            scaled.append(entry)

        return scaled
    
    def combine_scripts_to_ola_format(self, scripts):
        """
        Merge multiple scripts (each: [wait_ms, (fixture_id, rel_channel, value[, flag]), ..., wait_ms, ...])
        into (frame_delays_ms, dmx_frames) for OLA.
        Maps channel to absolute: abs_channel = self.fixture_addresses[fixture_id] + rel_channel
        """
        max_freq = 30.0
        min_frame_interval_ms = 33

        def script_event_iterator(script):
            idx = 0
            n = len(script)
            while idx < n:
                entry = script[idx]
                if not isinstance(entry, int):
                    idx += 1
                    continue

                wait_ms = max(0, entry)
                idx += 1

                commands = []
                while idx < n and not isinstance(script[idx], int):
                    cmd = script[idx]
                    if isinstance(cmd, tuple) and len(cmd) >= 3:
                        # Normalize to 3-tuple for OLA
                        commands.append((cmd[0], cmd[1], cmd[2]))
                    elif isinstance(cmd, list):
                        for sub in cmd:
                            if isinstance(sub, tuple) and len(sub) >= 3:
                                commands.append((sub[0], sub[1], sub[2]))
                    idx += 1

                yield wait_ms, commands

        all_scripts = [s for s in (scripts or []) if isinstance(s, list) and s]

        script_states = []
        for script in all_scripts:
            it = script_event_iterator(script)
            try:
                wait_ms, commands = next(it)
            except StopIteration:
                continue
            script_states.append({"iter": it, "next_wait": wait_ms, "pending": commands})

        if not script_states:
            return [0], [array.array("B", [0] * self.universe_size)]

        current_levels = [0] * self.universe_size
        dmx_frames = []
        frame_delays_ms = []

        while script_states:
            min_wait = min(s["next_wait"] for s in script_states)

            if min_wait == 0:
                ready = [s for s in script_states if s["next_wait"] == 0]
                for state in ready:
                    for fixture_id, rel_channel, value in state["pending"]:
                        abs_channel = self.fixture_addresses[fixture_id] + rel_channel
                        if isinstance(abs_channel, int) and 1 <= abs_channel <= self.universe_size:
                            current_levels[abs_channel - 1] = max(0, min(255, int(value)))
                    try:
                        wait_ms, commands = next(state["iter"])
                        state["next_wait"] = max(0, int(wait_ms))
                        state["pending"] = commands
                    except StopIteration:
                        script_states.remove(state)
                continue

            advance_ms = round(min_wait)

            dmx_frames.append(array.array("B", current_levels))
            frame_delays_ms.append(advance_ms)

            for state in script_states:
                state["next_wait"] = max(0, state["next_wait"] - advance_ms)

        blackout = array.array("B", [0] * self.universe_size)
        dmx_frames.append(blackout)
        frame_delays_ms.append(min_frame_interval_ms)

        result = {"frame_delays_ms": frame_delays_ms, "dmx_frames": dmx_frames}
        return result
    
    def convert_scripts_to_qlc_format(self, scripts, qlc_delay, qlc_lag, is_dimmer=False):
        qlc_delay = int(round(float(qlc_delay) * 1000))

        def _is_cmd_tuple(x):
            return isinstance(x, tuple) and len(x) == 3

        def _emit(entry, out, is_first_wait, lag_accum):
            if isinstance(entry, int):
                wait_time = entry
                if is_first_wait[0]:
                    wait_time += qlc_delay
                    is_first_wait[0] = False
                    out.append(self._qlc_wait(wait_time))
                elif is_dimmer:
                    scaled = wait_time * qlc_lag + lag_accum[0]
                    wait_scaled = int(scaled)
                    lag_accum[0] = scaled - wait_scaled
                    # Always emit at least 1 ms if original wait was > 0
                    if wait_time > 0 and wait_scaled == 0:
                        wait_scaled = 1
                        lag_accum[0] -= (1 - scaled)
                    out.append(self._qlc_wait(wait_scaled))
                else:
                    out.append(self._qlc_wait(wait_time))
            elif _is_cmd_tuple(entry) or (isinstance(entry, tuple) and len(entry) >= 4):
                # Accept tuples of length 3 or 4; ignore optional 4th (scale_dimmer)
                fixture, channel, value = entry[0], entry[1], entry[2]
                out.append(self._qlc_setfixture(fixture, channel, value, ""))
            elif isinstance(entry, list):
                for sub in entry:
                    _emit(sub, out, is_first_wait, lag_accum)
            elif isinstance(entry, str):
                out.append(entry)

        qlc_scripts = []
        for script in scripts:
            if not isinstance(script, list):
                continue
            script_copy = script.copy()
            out = []
            is_first_wait = [True]
            lag_accum = [0.0]  # Accumulate fractional ms across all waits
            for entry in script_copy:
                _emit(entry, out, is_first_wait, lag_accum)
            qlc_scripts.append(out)
        return qlc_scripts
            
class Queue:
    def __init__(self):
        self.queue = []

    def get_queue(self):
        return self.queue

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, item):
        self.queue.append(item)

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            return None

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            return None

    def edit(self, index, item):
        if index < len(self.queue):
            self.queue[index] = item
        else:
            print("Index out of range")

class Show:
    def __init__(self, name, struct, song_data):
        self.name = name
        self.struct = struct
        self.song_data = song_data
        self.bpm = struct["bpm"]
        self.wav_path = song_data["file"]
        self.beatinterval = 60 / (struct["bpm"])
