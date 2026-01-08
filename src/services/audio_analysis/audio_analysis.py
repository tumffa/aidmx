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
    alpha=0.2,                 # threshold factor vs segment-average (also used for song-level total RMS)
    active_frac_min=0.25,      # min fraction of frames above threshold to call an instrument "active"
    run_min_floor_sec=0.5,     # min duration for a quiet run to be eligible
    run_prominence_std=0.5,    # how many std above the mean run length counts as "clearly above"
):
    drum_rms = np.asarray(struct_data["drums_rms"], dtype=np.float64)
    bass_rms = np.asarray(struct_data["bass_rms"], dtype=np.float64)
    other_rms = np.asarray(struct_data["other_rms"], dtype=np.float64)
    vocals_rms = np.asarray(struct_data["vocals_rms"], dtype=np.float64)
    total_rms = np.asarray(struct_data["rms"], dtype=np.float64)  # song-level RMS

    L = min(len(drum_rms), len(bass_rms), len(other_rms), len(vocals_rms))
    if L == 0:
        return [{"pauses": []}]

    fps = int(struct_data.get("fps", 43))
    segments = struct_data.get("segments", []) or []

    def seg_idx(seg):
        s = max(0, int(round(seg["start"] * fps)))
        e = min(L, int(round(seg["end"] * fps)))
        return s, e

    def quiet_runs(mask, base_idx=0):
        # Returns [(start_abs, end_abs, length_frames), ...]
        if mask.size == 0:
            return []
        padded = np.pad(mask.astype(np.int8), (1, 1), mode='constant', constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        out = []
        for s, e in zip(starts, ends):
            if e > s:
                out.append((base_idx + int(s), base_idx + int(e), int(e - s)))
        return out

    def choose_long_quiet(runs, fps_local, min_floor_sec, prominence_std):
        # Accept runs clearly above average window time within this segment (or whole song for total_rms)
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

    pauses = []

    # Segment-wise processing
    for seg_idx_i, seg in enumerate(segments):
        s_idx, e_idx = seg_idx(seg)
        if e_idx - s_idx <= 0:
            continue

        # Segment averages and thresholds
        d_avg = float(np.mean(drum_rms[s_idx:e_idx])) or 1e-12
        b_avg = float(np.mean(bass_rms[s_idx:e_idx])) or 1e-12
        o_avg = float(np.mean(other_rms[s_idx:e_idx])) or 1e-12
        v_avg = float(np.mean(vocals_rms[s_idx:e_idx])) or 1e-12

        thr_d = alpha * d_avg
        thr_b = alpha * b_avg
        thr_o = alpha * o_avg
        thr_v = alpha * v_avg

        # Quiet masks within segment
        mask_d = (drum_rms[s_idx:e_idx] < thr_d)
        mask_b = (bass_rms[s_idx:e_idx] < thr_b)
        mask_o = (other_rms[s_idx:e_idx] < thr_o)
        mask_v = (vocals_rms[s_idx:e_idx] < thr_v)

        # Active in this segment?
        d_active = float(np.mean(~mask_d)) >= active_frac_min
        b_active = float(np.mean(~mask_b)) >= active_frac_min
        o_active = float(np.mean(~mask_o)) >= active_frac_min
        v_active = float(np.mean(~mask_v)) >= active_frac_min

        # Quiet runs and "clearly above average" selection
        runs_d = quiet_runs(mask_d, base_idx=s_idx)
        runs_b = quiet_runs(mask_b, base_idx=s_idx)
        runs_o = quiet_runs(mask_o, base_idx=s_idx)
        runs_v = quiet_runs(mask_v, base_idx=s_idx)

        long_d = choose_long_quiet(runs_d, fps, run_min_floor_sec, run_prominence_std)
        long_b = choose_long_quiet(runs_b, fps, run_min_floor_sec, run_prominence_std)
        long_o = choose_long_quiet(runs_o, fps, run_min_floor_sec, run_prominence_std)
        long_v = choose_long_quiet(runs_v, fps, run_min_floor_sec, run_prominence_std)

        # Ruling:
        # 1) If drums are active and there is a pause in drums, it is a pause.
        if d_active and long_d:
            for (ps, pe) in long_d:
                info = {"drums": True, "bass": False, "other": False, "vocals": False, "segment_index": seg_idx_i, "scale_ok": True}
                pauses.append((int(ps), int(pe), info))
            continue

        # 2) If drums are NOT active, but vocals/bass/other are:
        #    - If both bass and other are active, they both need to be paused (no overlap check). If both have any long quiet run,
        #      include all of their long quiet windows as pauses.
        #    - If there is only one instrument active among vocals/bass/other, that instrument alone being paused is enough.
        if not d_active:
            if b_active and o_active:
                if long_b and long_o:
                    for (ps, pe) in long_b:
                        info = {"drums": False, "bass": True, "other": False, "vocals": False, "segment_index": seg_idx_i, "scale_ok": True}
                        pauses.append((int(ps), int(pe), info))
                    for (ps, pe) in long_o:
                        info = {"drums": False, "bass": False, "other": True, "vocals": False, "segment_index": seg_idx_i, "scale_ok": True}
                        pauses.append((int(ps), int(pe), info))
            else:
                active_list = [("bass", b_active, long_b), ("other", o_active, long_o), ("vocals", v_active, long_v)]
                only_one_active = [name for name, act, _ in active_list if act]
                if len(only_one_active) == 1:
                    name = only_one_active[0]
                    longs = dict(bass=long_b, other=long_o, vocals=long_v)[name]
                    for (ps, pe) in longs:
                        info = {
                            "drums": False,
                            "bass": (name == "bass"),
                            "other": (name == "other"),
                            "vocals": (name == "vocals"),
                            "segment_index": seg_idx_i,
                            "scale_ok": True,
                        }
                        pauses.append((int(ps), int(pe), info))

    # Song-level total RMS quiet windows: rms < alpha * average(total_rms) across the whole song
    if total_rms.size:
        song_avg = float(np.mean(total_rms)) or 1e-12
        thr_total = alpha * song_avg
        mask_total = (total_rms < thr_total)
        runs_total = quiet_runs(mask_total, base_idx=0)
        long_total = choose_long_quiet(runs_total, fps, run_min_floor_sec, run_prominence_std)
        for (ps, pe) in long_total:
            info = {
                "drums": False,
                "bass": False,
                "other": False,
                "vocals": False,
                "mix": True,
                "segment_index": -1,  # song-level
                "scale_ok": True,
            }
            pauses.append((int(ps), int(pe), info))

    # Merge overlapping/adjacent pauses (within 0.15s)
    merged_pauses = []
    merge_gap_frames = int(round(0.15 * fps))
    for (s, e, info) in sorted(pauses, key=lambda x: (x[0], x[1])):
        if not merged_pauses:
            merged_pauses.append([s, e, info])
            continue
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
    for pause in pauses_out:
        print(f"Pause {pause[0]/fps:.2f}s to {pause[1]/fps:.2f}s - drums:{pause[2].get('drums')} bass:{pause[2].get('bass')} other:{pause[2].get('other')} vocals:{pause[2].get('vocals')} segment_index:{pause[2].get('segment_index')} scale_ok:{pause[2].get('scale_ok')}")
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