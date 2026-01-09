import librosa
import numpy as np
from src.services.audio_analysis import drum_analysis_strobes
from src.services.audio_analysis.seperate_drums_larsnet import separate_drums_with_larsnet
from src.services.audio_analysis.light_strength_envelope import calculate_light_strength_envelope

def analyze_audio(song_data, struct_data):
    params = []
    merged = dict(struct_data) if struct_data else {}

    out = initialize_song_metrics(song_data, struct_data=merged)
    params += out
    for d in out:
        merged.update(d)

    out = struct_stats(song_data, name=song_data.get("name"), struct_data=merged)
    params += out
    for d in out:
        merged.update(d)

    out = get_pauses(s:=
        song_data.get("name"), merged)
    params += out
    for d in out:
        merged.update(d)

    out = segment(song_data.get("name"), struct_data=merged)
    params += out
    for d in out:
        merged.update(d)

    out = calculate_light_strength_envelope(song_data, struct_data=merged)
    params += out

    return params

def initialize_song_metrics(song_data, struct_data):
    """Initial analysis to get basic metrics like RMS and separated drum onsets."""
    print("------Initializing song audio metrics")

    rms = np.array(get_rms(song_data)[0], dtype=float)
    bass_rms = np.array(get_rms(song_data, category=["bass"])[0], dtype=float)
    drums_rms = np.array(get_rms(song_data, category=["drums"])[0], dtype=float)
    other_rms = np.array(get_rms(song_data, category=["other"])[0], dtype=float)
    vocals_rms = np.array(get_rms(song_data, category=["vocals"])[0], dtype=float)

    total_rms = float(np.mean(rms))
    bass_average = float(np.mean(bass_rms))
    drums_average = float(np.mean(drums_rms))
    other_average = float(np.mean(other_rms))
    vocals_average = float(np.mean(vocals_rms))

    # Get the separated drums using LarsNet
    kick_snare_toms, sr = seperate_kick_toms_snare(song_data)
    larsnet_drums = kick_snare_toms["kick"] + kick_snare_toms["snare"] + kick_snare_toms["toms"]
    onset_parts = drum_analysis_strobes.get_onset_parts(segments=struct_data["segments"], input=larsnet_drums, sr=sr)

    params=[
        {
            "total_rms": total_rms,
            "rms": rms.tolist(),
            "bass_rms": bass_rms.tolist(),
            "drums_rms": drums_rms.tolist(),
            "other_rms": other_rms.tolist(),
            "vocals_rms": vocals_rms.tolist(),
            "bass_average": bass_average,
            "drums_average": drums_average,
            "other_average": other_average,
            "vocals_average": vocals_average
        },
        {
            "onset_parts": onset_parts
        }
    ]
    print("------Audio metrics and strobe parts initialized")

    return params

def segment(name, struct_data):
    print(f"----Defining energetic sections")

    segments = struct_data['segments']
    # Copy each dict to ensure we always attach the flag
    updated_segments = [dict(seg) for seg in segments]
    for seg in updated_segments:
        seg["is_chorus_section"] = False

    added_indices = set()
    volume_threshold = struct_data["loud_sections_average"] * 0.12
    volume_threshold2 = struct_data["loud_sections_average2"] * 0.12

    for i, seg in enumerate(segments):
        if i in added_indices or seg['label'] == 'start':
            continue

        volume_difference = seg["avg_combined"] - struct_data["loud_sections_average"]
        volume_difference2 = seg["avg_volume"] - struct_data["loud_sections_average2"]
        over_threshold = (volume_difference >= -volume_threshold and volume_difference2 >= -volume_threshold2)

        if (seg["label"] in struct_data["loud_sections"] or seg["label"] in ["intro", "outro", "inst"]) and over_threshold:
            if i < len(segments) - 1:
                if segments[i + 1]["label"] != "inst" or segments[i + 1]["avg_combined"] < seg["avg_combined"] or seg["label"] == "inst":
                    updated_segments[i]["is_chorus_section"] = True
                    added_indices.add(i)
                    continue
            else:
                if seg["avg_volume"] / struct_data["average_rms"] > 0.8:
                    updated_segments[i]["is_chorus_section"] = True
                    added_indices.add(i)
                    continue

        elif seg["label"] in struct_data["average_volumes"]:
            base = struct_data["average_volumes"][seg["label"]][0]
            if base and (seg["avg_combined"] / base > 1.1) and over_threshold:
                updated_segments[i]["is_chorus_section"] = True
                added_indices.add(i)
                continue

        elif seg["label"] == "bridge" and over_threshold:
            if struct_data["broken_bridges"]:
                updated_segments[i]["is_chorus_section"] = True
                added_indices.add(i)
                continue
            k = 0
            idxs = []
            add = False
            while i + k < len(segments) and segments[i + k]["label"] == "bridge":
                idxs.append(i + k); k += 1
            if i + k < len(segments):
                nxt = segments[i + k]
                if ((nxt["avg_combined"] - struct_data["loud_sections_average"] <= -volume_threshold * 1.3) and
                    (nxt["avg_volume"] - struct_data["loud_sections_average2"] <= -volume_threshold2 * 1.3)):
                    add = True
            if add:
                for idx in idxs:
                    updated_segments[idx]["is_chorus_section"] = True
                    added_indices.add(idx)
                continue

        elif (seg["label"] == "solo" and
              seg["avg_combined"] - struct_data["loud_sections_average"] >= -volume_threshold * 0.5 and
              seg["avg_volume"] - struct_data["average_rms"] >= -volume_threshold * 0.5):
            updated_segments[i]["is_chorus_section"] = True
            added_indices.add(i)
            continue

    energetic_count = sum(1 for s in updated_segments if s.get("is_chorus_section"))
    print(f"----Found {energetic_count} energetic sections")
    return [{"segments": updated_segments}]

def seperate_kick_toms_snare(song_data):
    drums_path = f"{song_data['demixed']}/drums.wav"
    output_dir = song_data['demixed']
    output_dict, sr = separate_drums_with_larsnet(drums_path, output_dir)
    return output_dict, sr

def get_rms(song_data=None, category=None, path=None) -> tuple[np.ndarray, float]:
    # Load the segment data from the JSON file
    if path:
        data_path = path
    elif category:
        data_path = song_data['demixed']
    elif song_data:
        data_path = song_data['file']
    else:
        raise Exception("Invalid arguments")
    # Calculate the average intensity
    if category:
        rms_arrays = []
        for instru in category:
            path = f"{data_path}/{instru}.wav"
            data_y, sr = librosa.load(path)
            rms_arrays.append(librosa.feature.rms(y=data_y)[0])
        rms = np.sum(np.stack(rms_arrays), axis=0)
        average = np.mean(rms)
        return rms, average
    data_y, sr = librosa.load(data_path)
    rms = librosa.feature.rms(y=data_y)[0]
    average = np.mean(rms)
    return rms, average

def struct_stats(song_data, name=None, category=None, path=None, rms=False, params=[], struct_data=None):
    print("------Analysing segment wise stats")
    params = params
    broken_bridges = False

    # Ensure numpy arrays
    if rms is False:
        rms = np.asarray(struct_data['rms'], dtype=np.float64)
    else:
        rms = np.asarray(rms, dtype=np.float64)

    drums_rms = np.asarray(struct_data['drums_rms'], dtype=np.float64)
    bass_rms = np.asarray(struct_data['bass_rms'], dtype=np.float64)
    drums_rms = drums_rms + bass_rms

    segments = struct_data['segments']
    song_rms = np.asarray(struct_data['rms'], dtype=np.float64)

    total_volumes = []
    avg_volumes = []
    drum_volumes = []
    loud_sections = []
    focus = {}
    is_quiet = False
    labels = {}
    i = 0

    # Precompute frames-per-second factor
    fps = 43

    for segment in segments:
        if segment["end"] - segment["start"] <= 1.5:
            segment["avg_combined"] = 0
            segment['avg_volume'] = 0
            segment['avg_drums'] = 0
            i += 1
            continue
        if segment["label"] == "bridge" and segment["start"] < 100:
            broken_bridges = True
        labels[segment['label']] = True
        segment_start = segment['start']
        segment_end = segment['end']

        start_idx = int(segment_start * fps)
        end_idx = int(segment_end * fps)

        # RMS slice and pauses
        rms_slice = rms[start_idx:end_idx].copy()
        temp_avg = float(np.mean(rms_slice)) if rms_slice.size else 0.0
        pauses = get_pauses_for_segment(rms_slice, temp_avg * 0.2)
        rms_slice = np.asarray(get_modified_rms(song_data, name=name, rms=rms_slice, pauses=pauses), dtype=np.float64)
        rms_slice = rms_slice[rms_slice != 0]

        # Drums slice and pauses
        drums_slice = drums_rms[start_idx:end_idx].copy()
        temp_avg = float(np.mean(drums_slice)) if drums_slice.size else 0.0
        pauses = get_pauses_for_segment(drums_slice, temp_avg * 0.2)
        drums_slice = np.asarray(get_modified_rms(song_data, name=name, rms=drums_slice, pauses=pauses), dtype=np.float64)
        drums_slice = drums_slice[drums_slice != 0]

        # Song slice and pauses
        song_slice = song_rms[start_idx:end_idx].copy()
        temp_avg = float(np.mean(song_slice)) if song_slice.size else 0.0
        pauses = get_pauses_for_segment(song_slice, temp_avg * 0.2)
        song_slice = np.asarray(get_modified_rms(song_data, name=name, rms=song_slice, pauses=pauses), dtype=np.float64)
        song_slice = song_slice[song_slice != 0]

        average_rms = float(np.mean(rms_slice)) if rms_slice.size else 0.0
        average_drums = float(np.mean(drums_slice)) if drums_slice.size else 0.0
        average_total_rms = float(np.mean(song_slice)) if song_slice.size else 0.0

        segment["avg_combined"] = average_total_rms
        segment['avg_volume'] = average_rms
        segment['avg_drums'] = average_drums
        avg_volumes.append(average_rms)
        total_volumes.append(average_total_rms)
        drum_volumes.append(average_drums)
        i += 1

    song_average = float(np.mean(avg_volumes)) if avg_volumes else 0.0
    total_rms = float(np.mean(song_rms)) if song_rms.size else 0.0
    drum_average = float(np.mean(drum_volumes)) if drum_volumes else 0.0

    verses = [segment for segment in segments if segment['label'] == 'verse']
    choruses = [segment for segment in segments if segment['label'] == 'chorus']

    verses_avg = float(np.mean([s['avg_combined'] for s in verses])) if verses else 0.0
    verses_avg2 = float(np.mean([s['avg_volume'] for s in verses])) if verses else 0.0
    choruses_avg = float(np.mean([s['avg_combined'] for s in choruses])) if choruses else 0.0
    choruses_avg2 = float(np.mean([s['avg_volume'] for s in choruses])) if choruses else 0.0

    versesdrum_average = float(np.mean([s['avg_drums'] for s in verses])) if verses else 0.0
    chorusesdrum_average = float(np.mean([s['avg_drums'] for s in choruses])) if choruses else 0.0

    if "inst" in labels:
        inst = [segment for segment in segments if segment['label'] == 'inst']
        inst_average = float(np.mean([s['avg_combined'] for s in inst])) if inst else 0.0
        inst_average2 = float(np.mean([s['avg_volume'] for s in inst])) if inst else 0.0
        average_volumes = {"verse": (verses_avg, verses_avg2), "chorus": (choruses_avg, choruses_avg2), "inst": (inst_average, inst_average2)}
    else:
        average_volumes = {"verse": (verses_avg, verses_avg2), "chorus": (choruses_avg, choruses_avg2)}

    if choruses_avg > total_rms:
        loud_sections.append(("chorus", (choruses_avg, choruses_avg2)))
    if verses_avg > total_rms:
        loud_sections.append(("verse", (verses_avg, verses_avg2)))
    if "inst" in labels:
        inst = [segment for segment in segments if segment['label'] == 'inst']
        inst_average = float(np.mean([s['avg_combined'] for s in inst])) if inst else 0.0
        inst_average2 = float(np.mean([s['avg_volume'] for s in inst])) if inst else 0.0
        if inst_average > total_rms:
            loud_sections.append(("inst", (inst_average, inst_average2)))

    loud_sections_average = float(np.mean([sec[1][0] for sec in loud_sections])) if loud_sections else 0.0
    loud_sections_average2 = float(np.mean([sec[1][1] for sec in loud_sections])) if loud_sections else 0.0

    sorted_sections = sorted(loud_sections, key=lambda item: item[1][0], reverse=True)
    focuses = ["first", "second", "third"]
    for i in range(len(sorted_sections)):
        focus[focuses[i]] = sorted_sections[i][0]

    params.append({
        'segments': segments,
        'average_rms': song_average,
        "loud_sections": [section[0] for section in loud_sections],
        "focus": focus,
        "is_quiet": is_quiet,
        "loud_sections_average": loud_sections_average,
        "loud_sections_average2": loud_sections_average2,
        "average_volumes": average_volumes,
        "drum_average": drum_average,
        "broken_bridges": broken_bridges
    })
    return params

def merge_short_sections(sections):
    merged_sections = []
    temp_section = sections[0]

    for i in range(1, len(sections)):
        current_section = sections[i]

        # If the current section is short, extend the temp section
        if current_section['end'] - temp_section['start'] <= 400:
            temp_section['end'] = current_section['end']
        else:
            # If the current section is not short, add the temp section to the merged sections and start a new temp section
            merged_sections.append(temp_section)
            temp_section = current_section

    # If there is a temp section at the end, add it to the merged sections
    merged_sections.append(temp_section)

    return merged_sections

def merge_same_category_sections(sections):
    merged_sections = []
    temp_section = sections[0]

    for i in range(1, len(sections)):
        current_section = sections[i]

        # If the current section has the same category as the temp section, extend the temp section
        if current_section['category'] == temp_section['category']:
            temp_section['end'] = current_section['end']
        else:
            # If the current section has a different category, add the temp section to the merged sections and start a new temp section
            merged_sections.append(temp_section)
            temp_section = current_section

    # If there is a temp section at the end, add it to the merged sections
    merged_sections.append(temp_section)

    return merged_sections

def get_modified_rms(song_data, name, rms=False, category=None, pauses=False, struct_data=None):
    # Normalize inputs to numpy arrays
    if isinstance(rms, list):
        rms = np.asarray(rms, dtype=np.float64)
    elif isinstance(rms, np.ndarray):
        rms = rms.astype(np.float64, copy=True)
    else:
        rms, _ = get_rms(song_data, category=category)
        rms = np.asarray(rms, dtype=np.float64)

    if isinstance(pauses, list):
        pause_ranges = pauses
    else:
        # Expect caller to pass pauses via struct_data; fallback to none
        pause_ranges = []

    modified_rms = rms.copy()
    # Zero-out ranges via slicing; accept (start,end) or (start,end,info)
    for rng in pause_ranges:
        start = int(rng[0]); end = int(rng[1])
        modified_rms[start:end] = 0.0

    return modified_rms

def get_pauses(
    name,
    struct_data,
    # Instrument-specific tunables
    alpha_drums=0.2,
    alpha_bass=0.2,
    alpha_other=0.2,
    alpha_vocals=0.2,
    active_frac_min_drums=0.4,
    active_frac_min_bass=0.4,
    active_frac_min_other=0.4,
    active_frac_min_vocals=0.4,
    run_min_floor_sec_drums=0.5,
    run_min_floor_sec_bass=0.5,
    run_min_floor_sec_other=0.5,
    run_min_floor_sec_vocals=0.5,
    run_prominence_std_drums=0.5,
    run_prominence_std_bass=0.5,
    run_prominence_std_other=0.5,
    run_prominence_std_vocals=0.5,
    boundary_margin_sec=1.0,
    # Drum-activity gating for interior pauses
    drums_activity_ratio_min=0.2,          # min drums avg / total avg within segment to allow interior pauses
    high_activity_percentile_drums=20.0,   # percentile used to detect “active” drum frames
    interior_guard_sec=1.0                 # activity must exist within this window before and after a quiet run
):
    drum_rms = np.asarray(struct_data["drums_rms"], dtype=np.float64)
    bass_rms = np.asarray(struct_data.get("bass_rms", []), dtype=np.float64)
    other_rms = np.asarray(struct_data.get("other_rms", []), dtype=np.float64)
    vocals_rms = np.asarray(struct_data.get("vocals_rms", []), dtype=np.float64)
    total_rms = np.asarray(struct_data.get("rms", []), dtype=np.float64)

    # Use drums length as canonical; others may be empty
    L = drum_rms.size
    if L == 0:
        return [{"pauses": []}]

    fps = int(struct_data.get("fps", 43))
    segments = struct_data.get("segments", []) or []
    boundary_margin_frames = int(round(boundary_margin_sec * fps))
    guard_frames = int(round(interior_guard_sec * fps))

    def seg_idx(seg):
        s = max(0, int(round(seg["start"] * fps)))
        e = min(L, int(round(seg["end"] * fps)))
        return s, e

    def quiet_runs(mask, base_idx=0):
        if mask.size == 0:
            return []
        padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant", constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        out = []
        for s, e in zip(starts, ends):
            if e > s:
                out.append((base_idx + int(s), base_idx + int(e), int(e - s)))
        return out

    def choose_long_quiet(runs, fps_local, min_floor_sec, prominence_std):
        if not runs:
            return []
        durs = np.array([r[2] for r in runs], dtype=np.float64)
        mean = float(np.mean(durs))
        std = float(np.std(durs))
        min_floor = int(round(min_floor_sec * fps_local))
        if len(durs) == 1:
            thr = max(min_floor, int(round(0.5 * durs[0])))
        else:
            thr = max(int(round(mean + prominence_std * std)), min_floor)
        return [(s, e) for (s, e, d) in runs if d >= thr]

    def overlaps(a_s, a_e, b_s, b_e):
        return (a_s < b_e) and (a_e > b_s)

    pauses = []

    # Segment-wise processing
    for seg_idx_i, seg in enumerate(segments):
        s_idx, e_idx = seg_idx(seg)
        if e_idx - s_idx <= 0:
            continue

        drum_seg = drum_rms[s_idx:e_idx]
        total_seg = total_rms[s_idx:e_idx] if total_rms.size >= e_idx else np.array([], dtype=np.float64)

        # Drums thresholds per segment
        d_avg = float(np.mean(drum_seg)) or 1e-12
        thr_d = alpha_drums * d_avg

        # Drums quiet mask and activity
        mask_d = (drum_seg < thr_d)
        d_active = float(np.mean(~mask_d)) >= active_frac_min_drums

        # Additional gate: drums must be meaningfully present in the segment (ratio vs total)
        if total_seg.size:
            t_avg = float(np.mean(total_seg)) or 1e-12
            drums_ratio_ok = (d_avg / max(t_avg, 1e-12)) >= float(drums_activity_ratio_min)
        else:
            drums_ratio_ok = d_avg >= 1e-12  # fallback

        # Drums quiet runs and selection
        runs_d = quiet_runs(mask_d, base_idx=s_idx)
        long_d = choose_long_quiet(runs_d, fps, run_min_floor_sec_drums, run_prominence_std_drums)

        # Boundary regions (frames)
        start_boundary_s = max(s_idx - boundary_margin_frames, s_idx)  # clamp to segment start
        start_boundary_e = min(s_idx + boundary_margin_frames, e_idx)
        end_boundary_s = max(e_idx - boundary_margin_frames, s_idx)
        end_boundary_e = min(e_idx + boundary_margin_frames, e_idx)    # clamp to segment end

        # 1) Mid-segment pauses: ONLY drums quiet runs allowed, and only if drums are present before AND after the quiet run
        if d_active and drums_ratio_ok and long_d:
            # Use a higher activity threshold to confirm surrounding drum activity
            thr_high = float(np.percentile(drum_seg, high_activity_percentile_drums)) if drum_seg.size else thr_d
            for (ps, pe) in long_d:
                interior = (ps >= start_boundary_e) and (pe <= end_boundary_s)
                if not interior:
                    continue

                # Surrounding activity check (guard window on both sides)
                before_s = max(s_idx, ps - guard_frames)
                before_e = max(s_idx, ps)
                after_s = min(e_idx, pe)
                after_e = min(e_idx, pe + guard_frames)

                before_active = (before_e > before_s) and bool(np.any(drum_rms[before_s:before_e] > thr_high))
                after_active = (after_e > after_s) and bool(np.any(drum_rms[after_s:after_e] > thr_high))

                if before_active and after_active:
                    info = {"drums": True, "segment_index": seg_idx_i, "scale_ok": True}
                    pauses.append((int(ps), int(pe), info))
                # If drums are only active on one side (e.g., just at the end), do NOT mark interior pause

        # 2) Boundary pauses: apply old logic near boundaries (drums-only mid, multi-stem permitted at boundaries)
        if d_active and long_d:
            for (ps, pe) in long_d:
                if overlaps(ps, pe, start_boundary_s, start_boundary_e) or overlaps(ps, pe, end_boundary_s, end_boundary_e):
                    info = {"drums": True, "segment_index": seg_idx_i, "scale_ok": True}
                    pauses.append((int(ps), int(pe), info))

        # Non-drums near boundaries (instrument-specific tunables)
        def stem_long_quiet(stem_rms, alpha_i, active_frac_min_i, run_min_floor_sec_i, run_prominence_std_i):
            if stem_rms.size == 0:
                return [], False
            seg_slice = stem_rms[s_idx:e_idx]
            avg = float(np.mean(seg_slice)) or 1e-12
            thr = alpha_i * avg
            mask = (seg_slice < thr)
            active = float(np.mean(~mask)) >= active_frac_min_i
            runs = quiet_runs(mask, base_idx=s_idx)
            longs = choose_long_quiet(runs, fps, run_min_floor_sec_i, run_prominence_std_i)
            return longs, active

        longs_b, b_active = stem_long_quiet(bass_rms, alpha_bass, active_frac_min_bass, run_min_floor_sec_bass, run_prominence_std_bass)
        longs_o, o_active = stem_long_quiet(other_rms, alpha_other, active_frac_min_other, run_min_floor_sec_other, run_prominence_std_other)
        longs_v, v_active = stem_long_quiet(vocals_rms, alpha_vocals, active_frac_min_vocals, run_min_floor_sec_vocals, run_prominence_std_vocals)

        if not d_active:
            if b_active and o_active:
                for (ps, pe) in longs_b:
                    if overlaps(ps, pe, start_boundary_s, start_boundary_e) or overlaps(ps, pe, end_boundary_s, end_boundary_e):
                        info = {"drums": False, "bass": True, "other": False, "vocals": False, "segment_index": seg_idx_i, "scale_ok": True}
                        pauses.append((int(ps), int(pe), info))
                for (ps, pe) in longs_o:
                    if overlaps(ps, pe, start_boundary_s, start_boundary_e) or overlaps(ps, pe, end_boundary_s, end_boundary_e):
                        info = {"drums": False, "bass": False, "other": True, "vocals": False, "segment_index": seg_idx_i, "scale_ok": True}
                        pauses.append((int(ps), int(pe), info))
            else:
                active_list = [("bass", b_active, longs_b), ("other", o_active, longs_o), ("vocals", v_active, longs_v)]
                only_one_active = [name for name, act, _ in active_list if act]
                if len(only_one_active) == 1:
                    name = only_one_active[0]
                    longs = dict(bass=longs_b, other=longs_o, vocals=longs_v)[name]
                    for (ps, pe) in longs:
                        if overlaps(ps, pe, start_boundary_s, start_boundary_e) or overlaps(ps, pe, end_boundary_s, end_boundary_e):
                            info = {
                                "drums": False,
                                "bass": (name == "bass"),
                                "other": (name == "other"),
                                "vocals": (name == "vocals"),
                                "segment_index": seg_idx_i,
                                "scale_ok": True,
                            }
                            pauses.append((int(ps), int(pe), info))

    # Merge overlapping/adjacent pauses (within 0.15s)
    merged_pauses = []
    merge_gap_frames = int(round(0.15 * fps))
    for (s, e, info) in sorted(pauses, key=lambda x: (x[0], x[1])):
        if not merged_pauses:
            merged_pauses.append([s, e, info]); continue
        ps, pe, pinfo = merged_pauses[-1]
        if s <= pe + merge_gap_frames:
            merged_pauses[-1][1] = max(pe, e)
            merged_pauses[-1][2] = {
                "drums": bool(pinfo.get("drums") or info.get("drums")),
                "bass": bool(pinfo.get("bass") or info.get("bass")),
                "other": bool(pinfo.get("other") or info.get("other")),
                "vocals": bool(pinfo.get("vocals") or info.get("vocals")),
                "segment_index": pinfo.get("segment_index", info.get("segment_index")),
                "scale_ok": bool(pinfo.get("scale_ok", False) or info.get("scale_ok", False)),
            }
        else:
            merged_pauses.append([s, e, info])

    pauses_out = [(int(s), int(e), info) for (s, e, info) in merged_pauses]
    return [{"pauses": pauses_out}]

def get_pauses_for_segment(rms, threshold):
    rms = np.asarray(rms, dtype=np.float64)
    quiet = rms < threshold

    window_size = 50
    min_quiet_frames = 40

    counts = np.convolve(quiet.astype(np.int32), np.ones(window_size, dtype=np.int32), mode='valid')
    window_ok = (counts >= min_quiet_frames).astype(np.int8)

    # Explicit coverage without overshoot, then clamp to quiet mask
    coverage = np.zeros(len(rms), dtype=bool)
    for i, ok in enumerate(window_ok):
        if ok:
            coverage[i:i + window_size] = True
    coverage &= quiet

    padded = np.pad(coverage.astype(np.int8), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    quiet_ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return quiet_ranges