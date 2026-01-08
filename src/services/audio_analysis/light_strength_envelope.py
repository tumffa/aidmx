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

    # Pull pauses metadata (frames at 43 fps by default)
    pauses = struct_data.get("pauses", [])
    fps = int(struct_data.get("fps", 43))
    total_frames = len(struct_data.get("rms", [])) if isinstance(struct_data.get("rms", []), (list, np.ndarray)) else None

    # First pass: compute flow windows for each segment
    all_flow_windows = []
    for segment in segments:
        flow_windows = determine_best_flow_instrument_windows(struct_data, segment)
        all_flow_windows.append(flow_windows)

    # Global normalization for smooth cross-segment transitions
    global_max_score = max(
        [w.get("score", 0.0) for seg_windows in all_flow_windows for w in seg_windows] or [1e-12]
    ) or 1e-12

    # Second pass: build envelopes per segment with neighbor-aware smooth transitions
    for i, segment in enumerate(segments):
        light_envelope = calculate_drums_envelope(segment)
        refined = refine_envelope_with_pauses(
            envelope=light_envelope,
            segment=segment,
            pauses=pauses,
            fps=fps,
            total_frames=total_frames
        )
        segment.setdefault("drum_analysis", {})
        segment["drum_analysis"]["light_strength_envelope"] = refined

        seg_windows = all_flow_windows[i]
        prev_tail_score = None
        next_head_score = None
        if i > 0 and all_flow_windows[i - 1]:
            prev_tail_score = all_flow_windows[i - 1][-1].get("score", None)
        if i + 1 < len(all_flow_windows) and all_flow_windows[i + 1]:
            next_head_score = all_flow_windows[i + 1][0].get("score", None)

        flow_envelope = build_flow_envelope(
            struct_data=struct_data,
            segment=segment,
            flow_windows=seg_windows,
            resolution_ms=20,
            ramp_ms=250,
            baseline=0.0,
            global_max_score=global_max_score,
            prev_tail_score=prev_tail_score,
            next_head_score=next_head_score
        )
        segment["flow_windows"] = seg_windows
        segment["flow_envelope"] = flow_envelope

        # Apply flow envelope to gaps outside active ranges that are >= 4 seconds
        try:
            gap_min_sec = 4.0
            seg_start_abs = float(segment.get("start", 0.0))
            seg_end_abs = float(segment.get("end", seg_start_abs))
            seg_dur = max(0.0, seg_end_abs - seg_start_abs)

            ref_times = np.asarray(refined.get("times", []), dtype=np.float64)  # relative to seg start
            ref_vals = np.asarray(refined.get("values", []), dtype=np.float64)

            # Build union of active ranges from refined envelope (segment-relative seconds)
            active_ranges = sorted(
                [(float(r.get("start_ms", 0)) / 1000.0, float(r.get("end_ms", 0)) / 1000.0)
                 for r in (refined.get("active_ranges") or [])],
                key=lambda x: x[0]
            )

            # Compute gaps outside active ranges
            gaps = []
            prev_end = 0.0
            for (rs, re) in active_ranges:
                rs = max(0.0, min(seg_dur, rs))
                re = max(0.0, min(seg_dur, re))
                if rs - prev_end >= gap_min_sec:
                    gaps.append((prev_end, rs))
                prev_end = max(prev_end, re)
            if seg_dur - prev_end >= gap_min_sec:
                gaps.append((prev_end, seg_dur))

            # Overlay flow envelope on those gaps
            fe_times = np.asarray(flow_envelope.get("times", []), dtype=np.float64)
            fe_vals = np.asarray(flow_envelope.get("values", []), dtype=np.float64)
            new_flow_ranges_ms = []
            if ref_times.size and ref_vals.size and fe_times.size and fe_vals.size and gaps:
                fe_interp = np.interp(
                    ref_times,
                    fe_times,
                    fe_vals,
                    left=fe_vals[0],
                    right=fe_vals[-1]
                )
                mask = np.zeros_like(ref_times, dtype=bool)
                for gs, ge in gaps:
                    mask |= (ref_times >= gs) & (ref_times <= ge)

                # Do not overlay where pauses already zeroed the envelope
                paused_zero_mask = ref_vals <= 1e-9
                mask &= ~paused_zero_mask

                # Record gap ranges (unchanged)
                new_flow_ranges_ms = []
                for gs, ge in gaps:
                    s_ms = int(np.ceil(gs * 1000.0))
                    e_ms = int(np.floor(ge * 1000.0))
                    if e_ms > s_ms:
                        new_flow_ranges_ms.append((s_ms, e_ms))

                ref_vals[mask] = np.maximum(ref_vals[mask], fe_interp[mask])
                refined["values"] = ref_vals.tolist()

            # Merge new flow-applied ranges into active_ranges
            if new_flow_ranges_ms:
                def merge_ranges_ms(ranges_ms):
                    rng = [(int(s), int(e)) for s, e in ranges_ms if int(e) > int(s)]
                    if not rng:
                        return []
                    rng.sort(key=lambda x: x[0])
                    merged = [rng[0]]
                    for s, e in rng[1:]:
                        ps, pe = merged[-1]
                        if s <= pe:
                            merged[-1] = (ps, max(pe, e))
                        else:
                            merged.append((s, e))
                    return [{"start_ms": s, "end_ms": e} for s, e in merged]

                existing_ms = [(int(r.get("start_ms", 0)), int(r.get("end_ms", 0))) for r in (refined.get("active_ranges") or [])]
                refined["active_ranges"] = merge_ranges_ms(existing_ms + new_flow_ranges_ms)

            # Store back the updated refined envelope
            segment["drum_analysis"]["light_strength_envelope"] = refined
        
        except Exception as e:
            print(f"[flow-overlay] Warning: could not apply flow envelope to gaps in segment {i}: {e}")

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

def refine_envelope_with_pauses(envelope, segment, pauses, fps=43, total_frames=None, require_flags=False):
    times_rel = np.asarray(envelope["times"], dtype=np.float64)
    values = np.asarray(envelope["values"], dtype=np.float64)
    seg_start_abs = float(envelope.get("segment_start", segment.get("start", 0.0)))
    seg_end_abs = float(envelope.get("segment_end", segment.get("end", seg_start_abs)))

    # Absolute time grid for this segment
    times_abs = seg_start_abs + times_rel
    scale = np.ones_like(times_abs, dtype=np.float64)

    # Detect envelope peaks once (local maxima); threshold defines a "meaningful" peak
    peak_threshold = 0.2
    if values.size >= 3:
        peak_idx = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
        peak_times = times_abs[peak_idx]
        peak_vals = values[peak_idx]
    else:
        peak_idx = np.array([], dtype=int)
        peak_times = np.array([], dtype=np.float64)
        peak_vals = np.array([], dtype=np.float64)

    # Build accepted pauses overlapping the segment
    accepted_pauses = []
    for p in (pauses or []):
        start_f, end_f = int(p[0]), int(p[1])
        info = p[2] if (isinstance(p, (list, tuple)) and len(p) >= 3) else {}
        drums_quiet = bool(info.get("drums", True))  # default True if missing
        scale_ok = bool(info.get("scale_ok", True))  # default True if missing

        if require_flags and not (drums_quiet and scale_ok):
            continue

        ps = float(start_f) / fps
        pe = float(end_f) / fps
        if pe <= ps:
            continue

        # Overlap with the current segment
        s_abs = max(ps, seg_start_abs)
        e_abs = min(pe, seg_end_abs)
        if e_abs <= s_abs:
            continue

        # Skip pauses that contain a strong peak (optional gating)
        if peak_idx.size:
            has_peak = np.any((peak_times >= s_abs) & (peak_times <= e_abs) & (peak_vals > peak_threshold))
            if has_peak:
                continue

        accepted_pauses.append((s_abs, e_abs))

    # Apply scaling from accepted pauses: ramp-in to 0, hold at 0, optional ramp-out for smoothness
    boundary_secs = 1.5
    for (s_abs, e_abs) in accepted_pauses:
        is_boundary_sensitive = (
            (s_abs <= seg_start_abs <= e_abs) or
            (s_abs <= seg_end_abs <= e_abs) or
            (e_abs >= seg_start_abs - boundary_secs and s_abs <= seg_start_abs + boundary_secs) or
            (e_abs >= seg_end_abs - boundary_secs and s_abs <= seg_end_abs + boundary_secs)
        )

        dur = max(e_abs - s_abs, 1e-6)
        ramp_ratio = 0.12 if is_boundary_sensitive else 0.33
        ramp_len = max(dur * ramp_ratio, 0.05)  # at least 50 ms
        ramp_in_end = min(s_abs + ramp_len, e_abs)
        ramp_out_end = min(e_abs + ramp_len, seg_end_abs)

        # Index masks
        idx_in_ramp = np.where((times_abs >= s_abs) & (times_abs <= ramp_in_end))[0]
        idx_hold = np.where((times_abs > ramp_in_end) & (times_abs <= e_abs))[0]
        idx_out_ramp = np.where((times_abs > e_abs) & (times_abs <= ramp_out_end))[0]

        # If no hold portion exists (very short pause), force full pause span to 0
        if idx_hold.size == 0:
            idx_pause_full = np.where((times_abs >= s_abs) & (times_abs <= e_abs))[0]
            if idx_pause_full.size:
                scale[idx_pause_full] = 0.0
        else:
            # Ramp-in (1 -> down), smoother on boundaries
            if idx_in_ramp.size:
                prog = (times_abs[idx_in_ramp] - s_abs) / max(ramp_in_end - s_abs, 1e-12)
                cand = (1.0 - prog) ** 2.5 if is_boundary_sensitive else 0.5 * (1.0 + np.cos(np.pi * prog))
                scale[idx_in_ramp] = np.minimum(scale[idx_in_ramp], cand)

            # Hold (force to 0)
            scale[idx_hold] = 0.0

            # Optional ramp-out (0 -> 1) to avoid sharp edges after pause
            if idx_out_ramp.size:
                prog = (times_abs[idx_out_ramp] - e_abs) / max(ramp_out_end - e_abs, 1e-12)
                cand = 0.5 * (1.0 - np.cos(np.pi * prog))
                scale[idx_out_ramp] = np.minimum(scale[idx_out_ramp], cand)

    refined_values = values * scale

    # Merge active ranges (existing + accepted pauses)
    def merge_ranges(ranges_ms):
        rng = [(int(s), int(e)) for s, e in ranges_ms if int(e) > int(s)]
        if not rng:
            return []
        rng.sort(key=lambda x: x[0])
        merged = [rng[0]]
        for s, e in rng[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return [{"start_ms": s, "end_ms": e} for s, e in merged]

    existing_ranges = envelope.get("active_ranges", []) or []
    base_ranges_ms = [(int(r.get("start_ms", 0)), int(r.get("end_ms", 0))) for r in existing_ranges if r is not None]

    pause_ranges_ms = []
    for (s_abs, e_abs) in accepted_pauses:
        # Use floor for start and ceil for end to avoid collapsing tiny pauses
        s_ms = int(np.floor((s_abs - seg_start_abs) * 1000.0))
        e_ms = int(np.ceil((e_abs - seg_start_abs) * 1000.0))
        if e_ms <= s_ms:
            e_ms = s_ms + 1  # ensure non-zero width
        pause_ranges_ms.append((s_ms, e_ms))

    active_ranges = merge_ranges(base_ranges_ms + pause_ranges_ms)

    return {
        "times": envelope["times"],
        "values": refined_values.tolist(),
        "segment_start": seg_start_abs,
        "segment_end": seg_end_abs,
        "active_ranges": active_ranges,
    }

def determine_best_flow_instrument_windows(
    struct_data,
    segment,
    window_sec=4.0,                 # window length to assess flow
    hop_sec=2.0,                    # hop between windows
    smooth_kernel_sec=0.5,          # smoothing kernel size (seconds)
    min_points=6,                   # minimum samples per window to score
    instruments=("bass", "other", "vocals"),  # non-drum stems to consider
    stickiness_margin_ratio=0.25,   # require ≥25% relative improvement to switch leader (window level)
    stickiness_min_delta=0.02,      # and ≥0.02 absolute score delta (window level)
    tie_eps=1e-9,                   # tie tolerance for equal scores
    instrument_priority=("vocals", "other", "bass")  # deterministic tie-break order
):
    fps = int(struct_data.get("fps", 43))

    # Pull signals (RMS proxies for “flow”). Drums are excluded.
    stem_map = {
        "bass":    np.asarray(struct_data.get("bass_rms", []),   dtype=np.float64),
        "other":   np.asarray(struct_data.get("other_rms", []),  dtype=np.float64),
        "vocals":  np.asarray(struct_data.get("vocals_rms", []), dtype=np.float64),
    }
    L = min([len(stem_map[k]) for k in instruments] or [0])
    if L == 0 or not segment:
        return []

    s_abs = float(segment.get("start", 0.0))
    e_abs = float(segment.get("end", s_abs))
    s_idx = max(0, int(round(s_abs * fps)))
    e_idx = min(L, int(round(e_abs * fps)))
    if e_idx - s_idx < min_points:
        return []

    w_frames = max(1, int(round(window_sec * fps)))
    h_frames = max(1, int(round(hop_sec * fps)))
    k_len = max(3, (int(round(smooth_kernel_sec * fps)) | 1))  # odd length

    # Hann smoothing kernel (deterministic)
    if k_len <= 1:
        kernel = np.array([1.0], dtype=np.float64)
    else:
        kernel = np.hanning(k_len)
        kernel = kernel / max(np.sum(kernel), 1e-12)

    def flow_score(x):
        if x.size < min_points:
            return 0.0
        x = x.astype(np.float64)
        x = x - float(np.mean(x))
        xs = np.convolve(x, kernel, mode="same")
        mod = float(np.std(xs))  # noticeable fluctuation
        d1 = np.diff(xs)
        d2 = np.diff(d1)
        jerk = float(np.mean(np.abs(d2))) if d2.size else 0.0
        smooth = 1.0 / (jerk + 1e-6)
        return mod * smooth

    def choose_leader(scores_dict, prev_leader=None):
        ordered = sorted(
            scores_dict.items(),
            key=lambda kv: (-kv[1], instrument_priority.index(kv[0]) if kv[0] in instrument_priority else len(instrument_priority))
        )
        best_inst, best_score = ordered[0]
        if prev_leader in scores_dict:
            prev_score = float(scores_dict[prev_leader])
            rel_ok = best_score >= (prev_score * (1.0 + stickiness_margin_ratio))
            abs_ok = (best_score - prev_score) >= stickiness_min_delta
            very_weak_prev = prev_score <= 1e-12
            is_tie = abs(best_score - prev_score) <= tie_eps
            if (best_inst != prev_leader) and not (rel_ok and abs_ok) and not very_weak_prev:
                return prev_leader, prev_score if not is_tie else prev_score
        return best_inst, float(best_score)

    # Windowed analysis inside this segment
    window_leaders = []
    prev_leader = None
    start = s_idx
    while start < e_idx:
        end = min(e_idx, start + w_frames)
        if end - start < min_points:
            break

        scores = {name: flow_score(stem_map[name][start:end]) for name in instruments}
        chosen_inst, chosen_score = choose_leader(scores, prev_leader)
        window_leaders.append((start, end, chosen_inst, float(chosen_score)))
        prev_leader = chosen_inst

        if start + h_frames >= e_idx:
            break
        start += h_frames

    # Build a deterministic per-frame leader timeline for the segment
    inst_to_idx = {name: i for i, name in enumerate(instruments)}
    seg_len = e_idx - s_idx
    leader_idx = np.full(seg_len, -1, dtype=np.int32)
    leader_score = np.full(seg_len, -np.inf, dtype=np.float64)

    for (start, end, leader, score) in window_leaders:
        li = inst_to_idx.get(leader, -1)
        if li < 0:
            continue
        for i_abs in range(start, end):
            if i_abs < s_idx or i_abs >= e_idx:
                continue
            i = i_abs - s_idx
            curr = leader_score[i]
            rel_ok = score >= (curr * (1.0 + stickiness_margin_ratio)) if np.isfinite(curr) and curr > 0 else True
            abs_ok = (score - curr) >= stickiness_min_delta if np.isfinite(curr) else True
            if (li != leader_idx[i]) and not (rel_ok and abs_ok):
                continue
            if np.isfinite(curr) and abs(score - curr) <= tie_eps:
                curr_name = instruments[leader_idx[i]] if leader_idx[i] >= 0 else None
                curr_pri = instrument_priority.index(curr_name) if curr_name in instrument_priority else len(instrument_priority)
                new_pri = instrument_priority.index(leader) if leader in instrument_priority else len(instrument_priority)
                if curr_name is not None and curr_pri <= new_pri:
                    continue
            leader_idx[i] = li
            leader_score[i] = score

    # Merge per-frame timeline into non-overlapping windows
    out = []
    def flush_range(s_rel, e_rel, li):
        if li < 0 or e_rel <= s_rel:
            return
        out.append({
            "start": (s_idx + s_rel) / fps,
            "end": (s_idx + e_rel) / fps,
            "leader": instruments[li],
            "score": float(np.mean(leader_score[s_rel:e_rel])) if e_rel > s_rel else float(leader_score[s_rel]),
        })

    cur_li = -1
    cur_s = 0
    for i in range(seg_len):
        if leader_idx[i] != cur_li:
            flush_range(cur_s, i, cur_li)
            cur_li = leader_idx[i]
            cur_s = i
    flush_range(cur_s, seg_len, cur_li)
    return out

def build_flow_envelope(
    struct_data,
    segment,
    flow_windows,
    resolution_ms=20,
    ramp_ms=250,
    baseline=0.0,
    global_max_score=None,   # song-level normalization for smooth cross-segment transitions
    prev_tail_score=None,    # previous segment’s last window score (raw)
    next_head_score=None,    # next segment’s first window score (raw)
    instrument_smooth_sec=0.8,   # stronger smoothing for instrument RMS within segment
    norm_p_low=10.0,             # wider robust normalization window
    norm_p_high=90.0,
    amplitude_power=0.7,         # compress window amplitude differences (<1 compresses)
    instrument_series_power=0.8, # compress instrument series peaks
    envelope_smooth_sec=0.6,     # final envelope smoothing
    peak_compress_gamma=0.85     # compress final peaks to reduce differences
):
    import numpy as np

    fps = int(struct_data.get("fps", 43))
    if not segment:
        return {"times": [], "values": []}

    seg_start_abs = float(segment.get("start", 0.0))
    seg_end_abs = float(segment.get("end", seg_start_abs))
    if seg_end_abs <= seg_start_abs:
        return {"times": [], "values": []}

    # No windows: return baseline for the segment
    if not flow_windows:
        times_abs = np.arange(seg_start_abs, seg_end_abs + 1e-9, resolution_ms / 1000.0, dtype=np.float64)
        return {"times": (times_abs - seg_start_abs).tolist(), "values": [baseline] * len(times_abs)}

    # Prepare instruments RMS series
    stems = {
        "bass":   np.asarray(struct_data.get("bass_rms", []),   dtype=np.float64),
        "other":  np.asarray(struct_data.get("other_rms", []),  dtype=np.float64),
        "vocals": np.asarray(struct_data.get("vocals_rms", []), dtype=np.float64),
    }
    L = min(len(stems["bass"]), len(stems["other"]), len(stems["vocals"])) if all(len(v) for v in stems.values()) else 0
    if L == 0:
        times_abs = np.arange(seg_start_abs, seg_end_abs + 1e-9, resolution_ms / 1000.0, dtype=np.float64)
        return {"times": (times_abs - seg_start_abs).tolist(), "values": [baseline] * len(times_abs)}

    # Sampling grid within the segment (absolute), output will be relative
    resolution_sec = resolution_ms / 1000.0
    ramp_sec = max(1e-3, ramp_ms / 1000.0)
    times_abs = np.arange(seg_start_abs, seg_end_abs + 1e-9, resolution_sec, dtype=np.float64)
    values = np.full_like(times_abs, baseline, dtype=np.float64)

    # Instrument interpolation at the segment grid
    song_times = np.arange(L, dtype=np.float64) / fps

    def interp_and_smooth(name):
        raw = stems.get(name, None)
        if raw is None or raw.size == 0:
            return np.zeros_like(times_abs)
        # Interpolate instrument RMS to the segment sampling grid
        series = np.interp(times_abs, song_times[:raw.size], raw[:raw.size], left=raw[0], right=raw[-1])
        # Smooth with Hann kernel over instrument_smooth_sec
        k = max(3, int(round(instrument_smooth_sec / resolution_sec)) | 1)
        if k > 3:
            kernel = np.hanning(k)
            kernel /= max(kernel.sum(), 1e-12)
            series = np.convolve(series, kernel, mode="same")
        # Percentile normalization to [0,1] within the segment
        p_low = float(np.percentile(series, norm_p_low))
        p_high = float(np.percentile(series, norm_p_high))
        scale = max(p_high - p_low, 1e-12)
        series = (series - p_low) / scale
        series = np.clip(series, 0.0, 1.0)
        # Compress peaks to reduce differences
        series = np.power(series, instrument_series_power)
        return series

    inst_series = {
        "bass": interp_and_smooth("bass"),
        "other": interp_and_smooth("other"),
        "vocals": interp_and_smooth("vocals"),
    }

    # Normalize window scores for amplitude (compressed)
    windows = sorted(flow_windows, key=lambda w: (w["start"], w["end"]))
    max_score = float(global_max_score if global_max_score is not None else (max(w.get("score", 0.0) for w in windows) or 1e-12))
    for w in windows:
        score_norm = max(0.0, min(1.0, float(w.get("score", 0.0)) / max_score))
        w["_amp"] = np.power(score_norm, float(amplitude_power))

    def smoothstep_cos(v0, v1, prog):
        # Vectorized: supports scalar or array inputs
        prog = np.clip(prog, 0.0, 1.0)
        return v0 + (v1 - v0) * 0.5 * (1.0 - np.cos(np.pi * prog))

    # Build envelope: per-window instrument contribution with cosine ramp-in/out, max-combined
    for j, w in enumerate(windows):
        ws = float(w["start"])
        we = float(w["end"])
        leader = w.get("leader", None)
        amp = float(w.get("_amp", 0.0))
        if leader not in inst_series or amp <= 0.0:
            continue

        # Masks on the segment grid
        in_win_mask = (times_abs >= ws) & (times_abs <= we)
        if not np.any(in_win_mask):
            continue

        ramp_in_mask = (times_abs >= ws) & (times_abs < ws + ramp_sec)
        ramp_out_mask = (times_abs > we - ramp_sec) & (times_abs <= we)
        hold_mask = in_win_mask & ~ramp_in_mask & ~ramp_out_mask

        # Convert masks to index arrays to avoid boolean shape mismatches
        idx_ramp_in = np.where(ramp_in_mask)[0]
        idx_ramp_out = np.where(ramp_out_mask)[0]
        idx_hold = np.where(hold_mask)[0]

        contrib = np.zeros_like(values)
        series = inst_series[leader] * amp  # same length as times_abs

        # Ramp-in from baseline (flowy cosine)
        if idx_ramp_in.size:
            prog = (times_abs[idx_ramp_in] - ws) / max(ramp_sec, 1e-12)
            target = series[idx_ramp_in]
            contrib[idx_ramp_in] = smoothstep_cos(baseline, target, prog)

        # Hold (pure instrument series)
        if idx_hold.size:
            contrib[idx_hold] = series[idx_hold]

        # Ramp-out to baseline (flowy cosine)
        if idx_ramp_out.size:
            prog = (we - times_abs[idx_ramp_out]) / max(ramp_sec, 1e-12)
            target = series[idx_ramp_out]
            contrib[idx_ramp_out] = smoothstep_cos(baseline, target, prog)

        # Max-combine contribution with existing envelope
        values = np.maximum(values, contrib)

        # Cross-window smoothing across gaps: glide from end of this window to start of next window
        if j + 1 < len(windows):
            next_ws = float(windows[j + 1]["start"])
            next_leader = windows[j + 1].get("leader", None)
            next_amp = float(windows[j + 1].get("_amp", 0.0))
            gap = max(next_ws - we, 0.0)
            if gap > 0:
                gap_mask = (times_abs > we) & (times_abs < next_ws)
                idx_gap = np.where(gap_mask)[0]
                if idx_gap.size:
                    denom = max(gap, ramp_sec)
                    prog = (times_abs[idx_gap] - we) / denom
                    # Levels at boundaries using interpolation on the same grid
                    end_level = np.interp(we, times_abs, series)
                    next_series = inst_series.get(next_leader, np.zeros_like(values)) * next_amp if next_leader in inst_series else np.zeros_like(values)
                    next_start_level = np.interp(next_ws, times_abs, next_series)
                    glide = smoothstep_cos(end_level, next_start_level, prog)
                    values[idx_gap] = np.maximum(values[idx_gap], glide)

    # Final envelope smoothing and peak compression
    k2 = max(3, int(round(envelope_smooth_sec / resolution_sec)) | 1)
    if k2 > 3:
        kernel2 = np.hanning(k2)
        kernel2 /= max(kernel2.sum(), 1e-12)
        values = np.convolve(values, kernel2, mode="same")
    values = np.power(np.clip(values, 0.0, 1.0), float(peak_compress_gamma))

    return {"times": (times_abs - seg_start_abs).tolist(), "values": values.tolist()}

def plot_light_envelope(envelope, flow_envelope=None, segment_title=None, show=True, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    times = np.asarray(envelope.get("times", []), dtype=np.float64)
    vals = np.asarray(envelope.get("values", []), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, vals, label="Drum+Pause envelope", color="#1f77b4")

    if flow_envelope:
        ft = np.asarray(flow_envelope.get("times", []), dtype=np.float64)
        fv = np.asarray(flow_envelope.get("values", []), dtype=np.float64)
        ax.plot(ft, fv, label="Flow envelope", color="#ff7f0e", alpha=0.7)

    # Shade active ranges (segment-relative seconds)
    for i, r in enumerate(envelope.get("active_ranges", []) or []):
        s = float(r.get("start_ms", 0)) / 1000.0
        e = float(r.get("end_ms", 0)) / 1000.0
        ax.axvspan(s, e, color="green", alpha=0.12, label="Active ranges" if i == 0 else None)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    if segment_title:
        ax.set_title(segment_title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)