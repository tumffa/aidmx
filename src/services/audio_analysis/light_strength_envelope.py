import numpy as np
from src.services.audio_analysis import drum_analysis_beats

def calculate_light_strength_envelope(song_data, struct_data):
    demix_path = song_data.get("demixed", None)
    beats = struct_data.get("beats", [])
    segments = struct_data.get("segments", [])

    drum_analysis_beats.analyze_drum_beat_pattern(
        demix_path=demix_path, 
        beats=beats, 
        segments=segments)
    
    # Pull silence metadata (frames at 43 fps by default)
    pauses = struct_data.get("pauses", [])
    silent_ranges = struct_data.get("silent_ranges", struct_data.get("quiet_ranges", []))
    fps = int(struct_data.get("fps", 43))
    total_frames = len(struct_data.get("rms", [])) if isinstance(struct_data.get("rms", []), (list, np.ndarray)) else None

    for segment in segments:
        light_envelope = calculate_drums_envelope(segment)
        # Refine with pauses and silent ranges
        refined = refine_envelope_with_pauses(
            envelope=light_envelope,
            segment=segment,
            pauses=pauses,
            silent_ranges=silent_ranges,
            fps=fps,
            total_frames=total_frames
        )
        segment["drum_analysis"]["light_strength_envelope"] = refined

    print(f"--------Calculated dimmer scaling function for {len(segments)} segments ({segments[0]['start']:.2f}s to {segments[-1]['end']:.2f}s)")
    return [{"segments": segments}]

def calculate_drums_envelope(segment, 
                             resolution_ms=5, 
                             min_strength=0.01, 
                             snare_multi=1, 
                             max_snare_fadeout=1.5, 
                             min_snare_fadeout=0.25,
                             kick_multi=0.5, 
                             max_kick_fadeout=0.8,
                             min_kick_fadeout=0.15):
    """
    Calculate a combined strength envelope from kick and snare beat-defining hits.
    Uses ONLY real hits (no phantom/grid markers) for more authentic light patterns.
    - Snare hits reach 100% intensity with longer fadeouts (up to max_snare_fadeout), at least min_snare_fadeout when possible
    - Kick hits reach 80% intensity with medium fadeouts (up to max_kick_fadeout), at least min_kick_fadeout when possible
    - All fadeouts complete to min_strength before next hit
    Times in the output are relative to segment start.
    """
    if not segment or "drum_analysis" not in segment:
        return {"times": [], "values": [], "segment_start": 0, "segment_end": 0, "active_ranges": []}
    
    drum_analysis = segment["drum_analysis"]
    segment_start = segment["start"]
    segment_end = segment["end"]
    segment_duration = segment_end - segment_start
    
    # Get kick and snare beat defining hits
    kick_hits = []
    kick_original_hits = []
    if "kick" in drum_analysis:
        if "beat_defining_hits" in drum_analysis["kick"]:
            kick_hits = drum_analysis["kick"]["beat_defining_hits"]
        if "hit_times_with_strength" in drum_analysis["kick"]:
            kick_original_hits = drum_analysis["kick"]["hit_times_with_strength"] 
    
    snare_hits = []
    snare_original_hits = []
    if "snare" in drum_analysis:
        if "beat_defining_hits" in drum_analysis["snare"]:
            snare_hits = drum_analysis["snare"]["beat_defining_hits"]
        if "hit_times_with_strength" in drum_analysis["snare"]:
            snare_original_hits = drum_analysis["snare"]["hit_times_with_strength"]
    
    # Create sets of actual hit times for fast lookup
    real_kick_times = {time for time, _ in kick_original_hits}
    real_snare_times = {time for time, _ in snare_original_hits}
    
    # If no hits found, return a flat baseline
    if not kick_hits and not snare_hits:
        return {
            "times": [0, segment_duration],
            "values": [min_strength, min_strength],
            "segment_start": segment_start,
            "segment_end": segment_end,
            "active_ranges": []
        }
    
    # Combine all hit times to find next hits
    all_hits = [(time, "snare") for time, _ in snare_hits if any(abs(time - rt) < 0.05 for rt in real_snare_times)]
    all_hits.extend([(time, "kick") for time, _ in kick_hits if any(abs(time - rt) < 0.05 for rt in real_kick_times)])
    all_hits.sort(key=lambda x: x[0])  # Sort by time
    
    # Process snare hits to calculate their fade-out windows
    snare_fadeouts = []
    kick_fadeouts = []
    
    # Only create fade-outs for actual snare hits
    for time, strength in snare_hits:
        # Check if this is a real snare hit
        if any(abs(time - rt) < 0.05 for rt in real_snare_times):
            # Find time to next defining hit (of any type)
            next_hit_time = segment_end
            for hit_time, _ in all_hits:
                if hit_time > time and hit_time < next_hit_time:
                    next_hit_time = hit_time
            
            # Calculate fade-out time with min/max bounds, but never beyond next hit
            time_to_next = next_hit_time - time
            # ...existing code...
            if time_to_next >= min_snare_fadeout:
                fadeout_time = min(time_to_next, max_snare_fadeout)
            else:
                fadeout_time = time_to_next  # must complete before next hit if too close
            
            # Store snare with its fadeout time
            snare_fadeouts.append((time, strength, fadeout_time))

    # Only create fade-outs for actual kick hits
    for time, strength in kick_hits:
        if any(abs(time - rt) < 0.05 for rt in real_kick_times):
            next_hit_time = segment_end
            for hit_time, _ in all_hits:
                if hit_time > time and hit_time < next_hit_time:
                    next_hit_time = hit_time

            time_to_next = next_hit_time - time
            # ...existing code...
            if time_to_next >= min_kick_fadeout:
                fadeout_time = min(time_to_next, max_kick_fadeout)
            else:
                fadeout_time = time_to_next
            
            kick_fadeouts.append((time, strength, fadeout_time))
    
    # Generate envelope points
    resolution_sec = resolution_ms / 1000
    envelope_times = []
    envelope_values = []

    # Start at segment start; no pre-appended duplicate sample
    current_time = segment_start

    # Add points for each time step
    while current_time <= segment_end:
        # Baseline value
        current_value = min_strength

        # Process kick contribution - directional immediate influence (post-hit only)
        kick_contribution = min_strength
        window_kick = 0.1  # 100ms immediate influence window
        for kick_time, kick_strength in kick_hits:
            if any(abs(kick_time - rt) < 0.05 for rt in real_kick_times):
                dt = current_time - kick_time
                if 0.0 <= dt <= window_kick:
                    time_weight = 1.0 - (dt / window_kick)
                    kick_value = kick_multi * time_weight
                    kick_contribution = max(kick_contribution, kick_value)

        # Extended kick fadeouts (unchanged directional)
        for kick_time, kick_strength, fadeout_time in kick_fadeouts:
            if kick_time <= current_time <= (kick_time + fadeout_time):
                fadeout_progress = (current_time - kick_time) / max(fadeout_time, 1e-12)
                if fadeout_progress >= 0.99:
                    kick_value = min_strength
                else:
                    decay_factor = (1.0 - fadeout_progress) ** 1.5
                    initial_strength = kick_multi
                    kick_value = min_strength + (initial_strength - min_strength) * decay_factor
                kick_contribution = max(kick_contribution, kick_value)

        # Process snare contribution - directional immediate influence (post-hit only)
        snare_contribution = min_strength
        window_snare_immediate = 0.1
        for snare_time, snare_strength in snare_hits:
            if any(abs(snare_time - rt) < 0.05 for rt in real_snare_times):
                dt = current_time - snare_time
                if 0.0 <= dt <= window_snare_immediate:
                    time_weight = 1.0 - (dt / window_snare_immediate)
                    immediate_value = snare_multi * time_weight
                    snare_contribution = max(snare_contribution, immediate_value)

        # Extended snare fadeouts (unchanged directional)
        for snare_time, snare_strength, fadeout_time in snare_fadeouts:
            if snare_time <= current_time <= (snare_time + fadeout_time):
                fadeout_progress = (current_time - snare_time) / max(fadeout_time, 1e-12)
                if fadeout_progress >= 0.99:
                    snare_value = min_strength
                else:
                    decay_factor = (1.0 - fadeout_progress) ** 1.5
                    snare_value = min_strength + (snare_multi - min_strength) * decay_factor
                snare_contribution = max(snare_contribution, snare_value)

        # Max of contributions
        current_value = max(kick_contribution, snare_contribution)

        envelope_times.append(current_time)
        envelope_values.append(current_value)

        current_time += resolution_sec

    # Add final point if needed
    if envelope_times and envelope_times[-1] < segment_end:
        envelope_times.append(segment_end)
        envelope_values.append(min_strength)
    
    # Add final point if needed
    if envelope_times[-1] < segment_end:
        envelope_times.append(segment_end)
        envelope_values.append(min_strength)

    # Convert to numpy arrays for faster operations
    times_arr = np.asarray(envelope_times, dtype=np.float64)
    values_arr = np.asarray(envelope_values, dtype=np.float64)

    # Find active ranges where envelope exceeds baseline (0.2)
    active_ranges = []
    in_active_range = False
    range_start = None

    for i, (time, value) in enumerate(zip(envelope_times, envelope_values)):
        # Detect transition into active range
        if value > 0.2 and not in_active_range:
            in_active_range = True
            range_start = time

        # Detect transition out of active range - now checks for exactly min_strength
        elif (value <= min_strength + 0.001 or i == len(envelope_values) - 1) and in_active_range:
            in_active_range = False
            range_end = time

            # Use numpy searchsorted + max over slice (faster than list.index + max)
            start_idx = int(np.searchsorted(times_arr, range_start, side='left'))
            end_idx = int(np.searchsorted(times_arr, range_end, side='left'))
            # Include end index (matching previous +1)
            end_idx = min(end_idx + 1, len(values_arr))
            if start_idx < end_idx:
                max_val = float(values_arr[start_idx:end_idx].max())
            else:
                max_val = float(values_arr[start_idx])

            active_ranges.append({
                "start_ms": int((range_start - segment_start) * 1000),
                "end_ms": int((range_end - segment_start) * 1000),
                "duration_ms": int((range_end - range_start) * 1000),
                "max_value": max_val
            })

    # Convert envelope_times to be relative to segment_start using numpy
    relative_envelope_times = (times_arr - segment_start).tolist()

    return {
        "times": relative_envelope_times,
        "values": envelope_values,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "active_ranges": active_ranges
    }

def refine_envelope_with_pauses(envelope, segment, pauses, silent_ranges, fps=43, total_frames=None):
    times_rel = np.asarray(envelope["times"], dtype=np.float64)
    values = np.asarray(envelope["values"], dtype=np.float64)
    seg_start = float(envelope.get("segment_start", segment.get("start", 0.0)))

    times_abs = seg_start + times_rel
    scale = np.ones_like(times_abs, dtype=np.float64)

    # Handle pauses as (start,end) or (start,end,info)
    boundary_secs = 1.5
    seg_start_abs = float(segment.get("start", 0.0))
    seg_end_abs = float(segment.get("end", seg_start_abs))
    # Compute song ends (absolute seconds)
    song_start_abs = 0.0
    song_end_abs = (float(total_frames) / float(fps)) if (total_frames is not None and fps > 0) else None

    for p in (pauses or []):
        start_f, end_f = int(p[0]), int(p[1])
        ps = float(start_f) / fps
        pe = float(end_f) / fps
        if pe <= ps:
            continue

        # Detect boundary-straddling or proximity
        crosses_start = (ps <= seg_start_abs <= pe)
        crosses_end = (ps <= seg_end_abs <= pe)
        near_start = (pe >= seg_start_abs - boundary_secs) and (ps <= seg_start_abs + boundary_secs)
        near_end = (pe >= seg_end_abs - boundary_secs) and (ps <= seg_end_abs + boundary_secs)
        is_boundary_sensitive = crosses_start or crosses_end or near_start or near_end

        # Fast ramp fraction near boundaries, normal otherwise
        ramp_ratio = 0.12 if is_boundary_sensitive else 0.33
        ramp_end = ps + (pe - ps) * ramp_ratio
        ramp_end = min(ramp_end, pe)  # guard tiny pauses

        # Apply attenuation: ramp down early, then hold zero
        in_ramp = (times_abs >= ps) & (times_abs <= ramp_end)
        if in_ramp.any():
            prog = (times_abs[in_ramp] - ps) / max(ramp_end - ps, 1e-12)
            cand = (1.0 - prog) ** 2.5 if is_boundary_sensitive else np.cos(0.5 * np.pi * prog)
            scale[in_ramp] = np.minimum(scale[in_ramp], cand)

        in_hold = (times_abs > ramp_end) & (times_abs <= pe)
        if in_hold.any():
            scale[in_hold] = 0.0

    # Apply pause scaling first
    refined_values = values * scale

    # 2) Handle silent_ranges special cases: first and last ranges
    if silent_ranges:
        sranges = sorted(
            [(int(s[0]), int(s[1])) if isinstance(s, (list, tuple)) and len(s) > 2 else (int(s[0]), int(s[1])) for s in silent_ranges],
            key=lambda x: x[0]
        )

        # First silent range: fade into first envelope value after the range IF that value < 0.5
        first_s, first_e = sranges[0]
        ss = float(first_s) / fps
        se = float(first_e) / fps

        mask_first = (times_abs >= ss) & (times_abs <= se)
        if mask_first.any():
            post_candidates = np.where(times_abs > se)[0]
            if post_candidates.size > 0:
                post_idx = int(post_candidates[0])
                v_post = float(values[post_idx])  # use raw envelope (pre-pauses)
                if v_post < 0.5:
                    prog = (times_abs[mask_first] - ss) / max(se - ss, 1e-12)
                    eased = np.sin(0.5 * np.pi * prog)
                    refined_values[mask_first] = v_post * eased

        # Last silent range: full fade to zero if it reaches song end; otherwise reach zero before 50%
        last_s, last_e = sranges[-1]
        ss2 = float(last_s) / fps
        se2 = float(last_e) / fps

        is_end_of_song = (total_frames is not None) and (int(last_e) >= int(total_frames) - 1)
        if se2 > ss2:
            if is_end_of_song:
                mask_full = (times_abs >= ss2) & (times_abs <= se2)
                if mask_full.any():
                    prog = (times_abs[mask_full] - ss2) / max(se2 - ss2, 1e-12)
                    eased_down = np.cos(0.5 * np.pi * prog)
                    refined_values[mask_full] = refined_values[mask_full] * eased_down
                mask_after = (times_abs > se2)
                if mask_after.any():
                    refined_values[mask_after] = 0.0
            else:
                ramp_ratio = 0.45
                ramp_end = ss2 + ramp_ratio * (se2 - ss2)
                mask_ramp = (times_abs >= ss2) & (times_abs <= ramp_end)
                if mask_ramp.any():
                    prog = (times_abs[mask_ramp] - ss2) / max(ramp_end - ss2, 1e-12)
                    eased_down = np.cos(0.5 * np.pi * prog)
                    refined_values[mask_ramp] = refined_values[mask_ramp] * eased_down
                mask_after = (times_abs > ramp_end) & (times_abs <= se2)
                if mask_after.any():
                    refined_values[mask_after] = 0.0

    def merge_ranges(ranges_ms):
        rng = [(int(s), int(e)) for s, e in ranges_ms if int(e) > int(s)]
        if not rng:
            return []
        rng.sort(key=lambda x: x[0])
        merged = [rng[0]]
        for s, e in rng[1:]:
            ps, pe = merged[-1]
            if s <= pe:  # overlap/adjacent
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return [{"start_ms": s, "end_ms": e} for s, e in merged]

    existing_ranges = envelope.get("active_ranges", []) or []
    base_ranges_ms = [(int(r.get("start_ms", 0)), int(r.get("end_ms", 0))) for r in existing_ranges if r is not None]

    # Pauses -> active ranges (segment-local), using ceil for start and floor for end
    pause_ranges_ms = []
    for p in (pauses or []):
        start_f, end_f = int(p[0]), int(p[1])
        ps = float(start_f) / fps
        pe = float(end_f) / fps
        s_abs = max(ps, seg_start_abs)
        e_abs = min(pe, seg_end_abs)
        if e_abs > s_abs:
            s_ms = int(np.ceil((s_abs - seg_start_abs) * 1000.0))
            e_ms = int(np.floor((e_abs - seg_start_abs) * 1000.0))
            if e_ms > s_ms:
                pause_ranges_ms.append((s_ms, e_ms))

    # Silent ranges that touch song start/end -> active ranges (segment-local)
    silent_song_edges_ms = []
    for s in (silent_ranges or []):
        start_f = int(s[0]); end_f = int(s[1])
        ss_abs = float(start_f) / fps
        se_abs = float(end_f) / fps
        touches_start = (ss_abs <= song_start_abs + 1e-9)
        touches_end = (song_end_abs is not None) and (se_abs >= song_end_abs - 1e-9)
        if not (touches_start or touches_end):
            continue
        s_abs = max(ss_abs, seg_start_abs)
        e_abs = min(se_abs, seg_end_abs)
        if e_abs > s_abs:
            s_ms = int(np.ceil((s_abs - seg_start_abs) * 1000.0))
            e_ms = int(np.floor((e_abs - seg_start_abs) * 1000.0))
            if e_ms > s_ms:
                silent_song_edges_ms.append((s_ms, e_ms))

    active_ranges = merge_ranges(base_ranges_ms + pause_ranges_ms + silent_song_edges_ms)

    return {
        "times": envelope["times"],
        "values": refined_values.tolist(),
        "segment_start": envelope.get("segment_start", segment.get("start", 0.0)),
        "segment_end": envelope.get("segment_end", segment.get("end", 0.0)),
        "active_ranges": active_ranges,
    }