from typing import Any, Dict, List, Tuple, Optional, Union
import math


RGBColor = Dict[str, int]
CommandTuple = Tuple[int, int, int, str]  # (fixture_id, channel, value, label)


class FixturePattern:
    def __init__(self, name: str):
        self.name = name
        self.pattern: List[Union[float, Tuple[Any, ...]]] = []

    def define_pattern(self, sequence: List[Union[float, Tuple[Any, ...]]]) -> None:
        """
        Define a pattern as a flattened sequence of entries where each entry is
        either a numeric fraction or a command tuple. The expected shapes are:

        - fraction (float/int); e.g., 0.5
        - (target, val1)
        - (target, val1, (val2, fraction2))

        Notes:
        - fraction and fraction2 must be numeric (float/int) representing timing fractions
          relative to your upstream unit (e.g., a beat).
        - target is a string: 'dimmer', 'rgb', 'shutter', or 'strobe'.
        - val1 is the primary value (e.g., 255, 'red', 'open').
        - (val2, fraction2) is optional, representing a secondary value change with
          its own fractional duration.

        This function validates and normalizes types, storing the sequence in self.pattern.
        """
        normalized: List[Union[float, Tuple[Any, ...]]] = []
        for entry in sequence:
            # Numeric fraction
            if isinstance(entry, (int, float)):
                normalized.append(float(entry))
                continue

            # Command tuple
            if not isinstance(entry, tuple) or len(entry) not in (2, 3):
                raise ValueError("tuple command must have length 2 or 3")

            target = str(entry[0])
            val1 = entry[1]

            if len(entry) == 3:
                extra = entry[2]
                if not isinstance(extra, tuple) or len(extra) != 2:
                    raise ValueError("third element must be a (val2, fraction2) tuple")
                val2, fraction2 = extra
                if not isinstance(fraction2, (int, float)):
                    raise ValueError("fraction2 must be numeric")
                normalized.append((target, val1, (val2, float(fraction2))))
            else:
                normalized.append((target, val1))

        self.pattern = normalized

    def _transition_sequence(
        self,
        fixture: Dict[str, Any],
        target: str,
        val1: Any,
        val2: Any,
        total_ms: int,
        scale_dimmer: Optional[Any],
    ) -> List[Any]:
        """
        Produce an incremental sequence taking a channel from val1 to val2 over total_ms.
        Returns a list shaped like: [wait, cmd, wait, cmd, ...], starting with 0 and
        issuing step-wise value changes until val2 is reached. If mapping fails, returns [].
        """
        # RGB transitions for separate RGB fixtures
        if str(target).lower() == "rgb" and fixture.get("colortype") == "seperate":
            # Resolve start/end RGB values
            def _resolve_rgb(v: Any) -> Optional[Tuple[int, int, int]]:
                if isinstance(v, str):
                    m = self._rgb_map().get(v.lower())
                    if not m:
                        return None
                    return int(m["red"]), int(m["green"]), int(m["blue"])
                if isinstance(v, tuple) and len(v) == 3:
                    r, g, b = v
                    return int(r), int(g), int(b)
                return None

            start_rgb = _resolve_rgb(val1)
            end_rgb = _resolve_rgb(val2)
            if not start_rgb or not end_rgb:
                return []

            cc = fixture.get("colorchannels", {}) or {}
            rch, gch, bch = cc.get("red"), cc.get("green"), cc.get("blue")

            # Compute steps: unify per-channel timelines
            min_wait_ms = 33
            steps_by_time = max(1, int(total_ms // min_wait_ms))
            increment = 10
            deltas = (
                end_rgb[0] - start_rgb[0],
                end_rgb[1] - start_rgb[1],
                end_rgb[2] - start_rgb[2],
            )
            steps_by_value = max(1, max(int(math.ceil(abs(d) / float(increment))) for d in deltas))
            steps = max(1, min(steps_by_time, steps_by_value))
            step_wait = max(1, int(round(total_ms / float(steps))))

            seqs: List[List[Any]] = []
            def build_seq(ch_name: str, v_start: int, v_end: int) -> Optional[List[Any]]:
                # Use compile_for_fixture for all emitted commands
                start_cmd = self._compile_for_fixture(
                    fixture, ("rgb", (ch_name, int(v_start))), scale_dimmer
                )
                end_cmd = self._compile_for_fixture(
                    fixture, ("rgb", (ch_name, int(v_end))), scale_dimmer
                )
                if start_cmd is None or end_cmd is None:
                    return None
                if total_ms <= 1 or v_start == v_end:
                    return [0, start_cmd, max(1, int(round(total_ms))), end_cmd]
                delta = v_end - v_start
                seq: List[Any] = [0, start_cmd]
                for i in range(1, steps + 1):
                    vi = int(round(v_start + delta * (i / float(steps)))) if i < steps else int(v_end)
                    step_cmd = self._compile_for_fixture(
                        fixture, ("rgb", (ch_name, int(vi))), scale_dimmer
                    )
                    if step_cmd is not None:
                        seq.append(step_wait)
                        seq.append(step_cmd)
                return seq

            r_seq = build_seq("red", start_rgb[0], end_rgb[0])
            g_seq = build_seq("green", start_rgb[1], end_rgb[1])
            b_seq = build_seq("blue", start_rgb[2], end_rgb[2])
            for s in (r_seq, g_seq, b_seq):
                if s:
                    seqs.append(s)
            if not seqs:
                return []
            return self._combine_sequences(seqs)

        c1 = self._compile_for_fixture(fixture, (target, val1), scale_dimmer)
        c2 = self._compile_for_fixture(fixture, (target, val2), scale_dimmer)
        if c1 is None or c2 is None:
            return []

        fid1, ch1, v_start, _ = c1
        fid2, ch2, v_end, _ = c2
        if ch1 != ch2 or fid1 != fid2:
            return []

        v_start = int(v_start)
        v_end = int(v_end)
        if total_ms <= 1 or v_start == v_end:
            # Nothing to transition, emit start and end as a single step
            return [0, c1, max(1, int(round(total_ms))), c2]

        delta = v_end - v_start

        # Choose increment granularity and time slicing
        # Aim for at least ~30ms per step and avoid more steps than value delta
        min_wait_ms = 33
        steps_by_time = max(1, int(total_ms // min_wait_ms))
        # Use a value increment suited for DMX (10 gives smooth enough curve)
        increment = 10
        steps_by_value = max(1, int(math.ceil(abs(delta) / float(increment))))
        steps = max(1, min(steps_by_time, steps_by_value))

        step_wait = max(1, int(round(total_ms / float(steps))))

        seq: List[Any] = [0, c1]
        for i in range(1, steps + 1):
            # Linear interpolation
            if i < steps:
                vi = v_start + int(round(delta * (i / float(steps))))
            else:
                vi = v_end
            cmd_i = self._compile_for_fixture(fixture, (target, int(vi)), scale_dimmer)
            if cmd_i is not None:
                seq.append(step_wait)
                seq.append(cmd_i)

        return seq

    def _combine_sequences(self, sequences: List[List[Any]]) -> List[Any]:
        """
        Combine multiple [wait, cmd, wait, cmd, ...] sequences by time-slicing.
        Picks the shortest next wait, appends it and the corresponding command,
        subtracts from other sequences, and repeats until all sequences are consumed.
        """
        # Normalize: ensure each sequence starts with a wait int
        norm_seqs: List[List[Any]] = []
        for seq in sequences:
            if not seq:
                continue
            if isinstance(seq[0], int):
                norm_seqs.append(seq[:])
            else:
                # If the sequence unexpectedly starts with a command, prefix zero wait
                norm_seqs.append([0] + seq[:])

        # Pointers and current wait heads
        positions = [0 for _ in norm_seqs]
        waits: List[Optional[int]] = []
        for s in norm_seqs:
            waits.append(int(s[0]) if len(s) > 0 and isinstance(s[0], int) else None)

        combined: List[Any] = []

        def all_done() -> bool:
            for idx, pos in enumerate(positions):
                if pos < len(norm_seqs[idx]):
                    return False
            return True

        while not all_done():
            # Find minimal next wait among active sequences
            active_pairs = [
                (idx, waits[idx])
                for idx in range(len(norm_seqs))
                if positions[idx] < len(norm_seqs[idx]) and waits[idx] is not None
            ]
            if not active_pairs:
                break
            _, min_wait = min(active_pairs, key=lambda x: x[1])
            # Avoid emitting a leading 0ms wait; it's implicit at t=0
            if not (not combined and int(min_wait) == 0):
                combined.append(int(min_wait))
            # Subtract minimal from all waits
            for idx, w in enumerate(waits):
                if w is not None:
                    waits[idx] = max(0, w - int(min_wait))

            # Emit all commands that are now at zero wait
            zero_idxs = [
                idx for idx, w in enumerate(waits)
                if w is not None and w == 0 and positions[idx] < len(norm_seqs[idx])
            ]
            batch_cmds: List[Any] = []
            for emit_idx in zero_idxs:
                positions[emit_idx] += 1  # consume the 0-wait
                if positions[emit_idx] < len(norm_seqs[emit_idx]):
                    cmd = norm_seqs[emit_idx][positions[emit_idx]]
                    batch_cmds.append(cmd)
                    positions[emit_idx] += 1
                # Load next wait for this sequence
                if positions[emit_idx] < len(norm_seqs[emit_idx]) and isinstance(
                    norm_seqs[emit_idx][positions[emit_idx]], int
                ):
                    waits[emit_idx] = norm_seqs[emit_idx][positions[emit_idx]]
                else:
                    waits[emit_idx] = None

            if batch_cmds:
                if len(batch_cmds) == 1:
                    combined.append(batch_cmds[0])
                else:
                    combined.append(batch_cmds)

        return combined

    def compile_pattern(
        self,
        fixture: Dict[str, Any],
        interval: int,
        scale_dimmer: Optional[Any] = None,
    ) -> List[Any]:
        """
        Compile self.pattern into a list of wait_time and command tuples.
        Fractions become wait_time = fraction * interval.
        (target, val1) becomes a single command via _compile_for_fixture.
        (target, val1, (val2, fraction)) becomes a transition sequence over
        fraction * interval. Consecutive 3-tuples with different targets are
        converted then merged by time-slicing.
        """
        output: List[Any] = []
        i = 0
        n = len(self.pattern)
        while i < n:
            entry = self.pattern[i]
            # Numeric fraction
            if isinstance(entry, (int, float)):
                wait_ms = max(1, int(round(float(entry) * float(interval))))
                output.append(wait_ms)
                i += 1
                continue

            if isinstance(entry, tuple):
                # (target, val1)
                if len(entry) == 2:
                    target, val1 = entry
                    cmd = self._compile_for_fixture(fixture, (target, val1), scale_dimmer)
                    if cmd is not None:
                        output.append(cmd)
                    i += 1
                    continue

                # (target, val1, (val2, fraction)) possibly multiple consecutive
                if len(entry) == 3:
                    group_sequences: List[List[Any]] = []
                    used_targets: set = set()
                    j = i
                    while j < n and isinstance(self.pattern[j], tuple) and len(self.pattern[j]) == 3:
                        t, v1, extra = self.pattern[j]  # type: ignore
                        if not isinstance(extra, tuple) or len(extra) != 2:
                            break
                        v2, frac = extra
                        t_key = str(t).lower()
                        # Stop grouping if target repeats within the group
                        if t_key in used_targets:
                            break
                        used_targets.add(t_key)
                        total_ms = max(1, int(round(float(frac) * float(interval))))
                        seq = self._transition_sequence(
                            fixture, t_key, v1, v2, total_ms, scale_dimmer
                        )
                        if seq:
                            group_sequences.append(seq)
                        j += 1

                    if group_sequences:
                        combined = self._combine_sequences(group_sequences)
                        output.extend(combined)
                    i = j
                    continue

            # Fallback: unrecognized entry, skip
            i += 1

        return output

    @staticmethod
    def _rgb_map() -> Dict[str, RGBColor]:
        return {
            "white": {"red": 255, "green": 255, "blue": 255},
            "red": {"red": 255, "green": 0, "blue": 0},
            "green": {"red": 0, "green": 255, "blue": 0},
            "blue": {"red": 0, "green": 0, "blue": 255},
            "pink": {"red": 255, "green": 0, "blue": 255},
            "yellow": {"red": 255, "green": 255, "blue": 0},
            "cyan": {"red": 0, "green": 255, "blue": 255},
            "orange": {"red": 255, "green": 80, "blue": 0},
            "purple": {"red": 128, "green": 0, "blue": 255},
        }

    def _setfixture(
            self, fixture_id: int, channel: int, value: int, scale_dimmer: Optional[Any] = None) -> CommandTuple:
        """
        Return a 4-length command tuple compatible with the existing queue format.
        The scale_dimmer flag, when relevant, can be handled by downstream logic; the
        command tuple itself remains 4-length as requested.
        """
        return (int(fixture_id), int(channel), int(value), scale_dimmer)

    def _map_shutter_value(self, fixture: Dict[str, Any], value: Union[str, int]) -> int:
        if isinstance(value, int):
            return value
        shutters = fixture.get("shutters", {}) or {}
        if isinstance(value, str):
            key = value.lower()
            if key in shutters:
                return int(shutters[key])
        # fallback: open if available else pass-through
        return int(shutters.get("open", 255))

    def _map_strobe_value(self, fixture: Dict[str, Any], value: Union[str, int]) -> int:
        if isinstance(value, int):
            return value
        strobevalues = fixture.get("strobevalues", {}) or {}
        if isinstance(value, str):
            key = value.lower()
            if key in strobevalues:
                return int(strobevalues[key])
            if key == "nicestrobe":
                return int(strobevalues.get("nicestrobe", 255))
        return 255

    def _percent_to_dimmer_value(self, fixture: Dict[str, Any], percent: Union[int, float]) -> int:
        """
        Convert a dimmer percentage (0-100) to raw DMX value using the fixture's
        dimmer_range. Defaults to [0, 255] if not specified.
        
        Args:
            fixture: Fixture dictionary that may contain 'dimmer_range' key.
            percent: Dimmer percentage from 0 to 100.
        
        Returns:
            Raw DMX value as an integer.
        """
        dimmer_range = fixture.get("dimmer_range", [0, 255])
        if isinstance(dimmer_range, (list, tuple)) and len(dimmer_range) >= 2:
            min_val, max_val = int(dimmer_range[0]), int(dimmer_range[1])
        else:
            min_val, max_val = 0, 255
        
        # Clamp percent to 0-100
        percent = max(0, min(100, float(percent)))
        raw_value = min_val + (percent / 100.0) * (max_val - min_val)
        return int(round(raw_value))

    def _compile_for_fixture(
        self,
        fixture: Dict[str, Any],
        target_value: Tuple[str, Any],
        scale_dimmer: Optional[Any] = None,
    ) -> Optional[CommandTuple]:
        """
        Compile a single (target, value) command for a given fixture.

        Args:
            fixture: Fixture dictionary with channel metadata (expects keys like
                     'id', 'dimmer', 'shutter', 'strobe', 'colorchannels', 'colortype', etc.).
            target_value: Tuple of (target, value) where target is one of
                          'dimmer', 'rgb', 'shutter', 'strobe'.
            scale_dimmer: Optional flag carried in the 4-length tuple.

        Returns:
            A single 4-length command tuple (id, channel, value, scale_dimmer),
            or None if the mapping cannot produce a single channel command.
        """
        target, value = target_value
        tgt = str(target).lower()
        rgb_map = self._rgb_map()

        if tgt == "dimmer":
            chan = fixture.get("dimmer")
            if chan is not None:
                # Convert percentage (0-100) to raw DMX value using dimmer_range
                raw_value = self._percent_to_dimmer_value(fixture, value)
                return self._setfixture(fixture["id"], chan, raw_value, scale_dimmer)
            return None

        if tgt == "rgb":
            colortype = fixture.get("colortype")
            cc = fixture.get("colorchannels", {}) or {}
            if colortype == "single":
                # Support wheel-style single color channel via string name or int value
                cch_main = cc.get("single")
                if cch_main is None:
                    return None
                if isinstance(value, str):
                    color_key = value.lower()
                    colour_map = fixture.get("colourmap", {}) or {}
                    wheel_val = colour_map.get(color_key)
                    if wheel_val is None:
                        # fallback to direct rgb_map if no wheel mapping
                        rgb_vals = rgb_map.get(color_key)
                        if rgb_vals is None:
                            return None
                        # cannot compress 3 channels into single wheel without mapping
                        return None
                    return self._setfixture(fixture["id"], cch_main, int(wheel_val))
                elif isinstance(value, int):
                    return self._setfixture(fixture["id"], cch_main, int(value))
                # tuple for single wheel not supported as single command
                return None
            else:
                # 'seperate' RGB: allow single-channel update via (channel_name, value)
                if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                    ch_name, v = value[0].lower(), value[1]
                    ch_map = {
                        "red": cc.get("red"),
                        "green": cc.get("green"),
                        "blue": cc.get("blue"),
                    }
                    ch = ch_map.get(ch_name)
                    if ch is None:
                        return None
                    return self._setfixture(fixture["id"], int(ch), int(v))
                # other rgb value types are not single-channel commands
                return None

        if tgt == "shutter":
            chan = fixture.get("shutter")
            if chan is not None:
                v = self._map_shutter_value(fixture, value)
                return self._setfixture(fixture["id"], chan, v)
            return None

        if tgt == "strobe":
            chan = fixture.get("strobe")
            if chan is not None:
                v = self._map_strobe_value(fixture, value)
                return self._setfixture(fixture["id"], chan, v)
            return None

        return None


if __name__ == "__main__":
    # Example: compile a pattern for a single fixture
    fixture_example = {
        "address": 7,
        "colorchannels": {
            "blue": 2,
            "green": 1,
            "red": 0,
        },
        "colortype": "seperate",
        "dimmer": 3,
        "id": 1,
        "nicestrobe": 250,
        "shutter": 4,
        "shutters": {
            "open": 0,
        },
        "strobe": 4,
        "stroberange": [20, 255],
    }

    fp = FixturePattern(name="Example")
    fp.define_pattern([
        0.25,
        ("dimmer", 100),  # 100% = full brightness
        0.5,
        ("dimmer", 0, (100, 1)),  # fade from 0% to 100%
        ("rgb", "red", ("blue", 1)),
        0.5,
        ("dimmer", 0, (100, 1)),  # fade from 0% to 100%
    ])

    compiled = fp.compile_pattern(fixture_example, interval=500, scale_dimmer="both")
    print("compiled_pattern:", compiled)
