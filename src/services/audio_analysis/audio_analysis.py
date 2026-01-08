import librosa
import numpy as np
from src.services.audio_analysis import drum_analysis_strobes
from src.services.audio_analysis import drum_analysis_beats
from src.services.audio_analysis.seperate_drums_larsnet import separate_drums_with_larsnet

def analyze_audio(song_data, struct_data):
    params = []
    params += initialize_song_metrics(song_data, struct_data=struct_data)
    params += struct_stats(song_data, name=song_data.get("name"), struct_data=struct_data)
    params += segment(song_data.get("name"), struct_data=struct_data)
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

    enriched_segments = drum_analysis_beats.analyze_drum_beat_pattern(
        demix_path=song_data["demixed"], 
        beats=struct_data["beats"], 
        segments=struct_data["segments"])

    params=[
        {
            "segments": enriched_segments,
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
    print("------Audio metrics, strobe parts and dimmer scaling done")

    return params

def segment(name, struct_data):
    """Finds sections that are 'energetic' based on their label and volume. Also finds pauses in the song.

    Args:
        name (str): The name of the song
        struct_data (dict): struct data

    Returns:
        params (list): List with a dictionary containing the chorus sections and pauses
    """
    print(f"----Defining energetic sections and pauses")
    pauses, silent_ranges = get_pauses(name, struct_data)
    segments = struct_data['segments']
    chorus_sections = []
    added_sections = []
    i = 0
    volume_threshold = struct_data["loud_sections_average"] * 0.12
    volume_threshold2 = struct_data["loud_sections_average2"] * 0.12

    for segment in segments:
        segment_start = segment['start']

        if segment_start in added_sections:
            i+=1
            continue
        if segment['label'] == 'start':
            i+=1
            continue
        temp = {"seg_start": segment_start, "seg_end": segment["end"], "label": segment["label"], "avg_volume": segment["avg_volume"], "avg_combined": segment["avg_combined"]}
        if i > 0:
            if segments[i-1]["start"] not in added_sections and segments[i - 1]["label"] == "bridge":
                temp["after_bridge"] = True

        volume_difference = segment["avg_combined"]-struct_data["loud_sections_average"]
        volume_difference2 = segment["avg_volume"]-struct_data["loud_sections_average2"]
        over_threshold = (volume_difference >= -volume_threshold and volume_difference2 >= -volume_threshold2)
        if (segment["label"] in struct_data["loud_sections"] or segment["label"] in ["intro", "outro", "inst"]) and over_threshold:
            if i < len(segments) - 1:
                if (segments[i+1]["label"] != "inst"):
                    chorus_sections.append(temp)
                    added_sections.append(segment_start)
                    continue
                elif segments[i+1]["avg_combined"] < segment["avg_combined"]:
                    chorus_sections.append(temp)
                    added_sections.append(segment_start)
                    continue
                elif segment["label"] == "inst":
                    chorus_sections.append(temp)
                    added_sections.append(segment_start)
                    continue
            elif segment["avg_volume"]/struct_data["average_rms"] > 0.8:
                chorus_sections.append(temp)
                added_sections.append(segment_start)
                continue
        elif segment["label"] in struct_data["average_volumes"]:
            if segment["avg_combined"]/struct_data["average_volumes"][segment["label"]][0] > 1.1 and over_threshold:
                    chorus_sections.append(temp)
                    added_sections.append(segment_start)
                    continue
        elif segment["label"] == "bridge" and over_threshold:
            if struct_data["broken_bridges"] == True:
                chorus_sections.append(temp)
                added_sections.append(segment_start)
                continue
            k = 0
            indexes = []
            add = False
            while True:
                if segments[i + k]["label"] == "bridge":
                    indexes.append(i + k)
                    k += 1
                    continue
                if segments[i + k]["avg_combined"] - struct_data["loud_sections_average"] <= -volume_threshold*1.3 and segments[i + k]["avg_volume"] - struct_data["loud_sections_average2"] <= -volume_threshold2*1.3:
                    add = True
                break
            if add:
                for index in indexes:
                    temp = {"seg_start": segments[index]["start"], "seg_end": segments[index]["end"], "label": segments[index]["label"], "avg_volume": segments[index]["avg_volume"], "avg_combined": segments[index]["avg_combined"]}
                    chorus_sections.append(temp)
                    added_sections.append(segments[index]["start"])
                continue
        elif segment["label"] == "solo" and segment["avg_combined"] - struct_data["loud_sections_average"] >= -volume_threshold*0.5 and segment["avg_volume"] - struct_data["average_rms"] >= -volume_threshold*0.5:
            chorus_sections.append(temp)
            added_sections.append(segment_start)
            continue
        i += 1
    params=[{'chorus_sections': chorus_sections, "pauses": pauses, "silent_ranges": silent_ranges}]
    print(f"----Found {len(chorus_sections)} energetic sections and {len(pauses)} pauses")
    return params

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
        pause_ranges, _ = get_pauses(name, struct_data)

    modified_rms = rms.copy()
    # Zero-out ranges via slicing
    for start, end in pause_ranges:
        modified_rms[start:end] = 0.0
    
    return modified_rms

def get_pauses(name, struct_data):
    drum_rms = np.asarray(struct_data["drums_rms"], dtype=np.float64)
    bass_rms = np.asarray(struct_data["bass_rms"], dtype=np.float64)
    other_rms = np.asarray(struct_data["other_rms"], dtype=np.float64)
    vocals_rms = np.asarray(struct_data["vocals_rms"], dtype=np.float64)
    rms = np.asarray(struct_data["rms"], dtype=np.float64)

    average_volume = struct_data["total_rms"]
    quiet_threshold_drums = 0.2 * struct_data["drums_average"]
    quiet_threshold_bass = 0.2 * struct_data["bass_average"]
    quiet_threshold_other = 0.2 * struct_data["other_average"]
    quiet_threshold_vocals = 0.2 * struct_data["vocals_average"]
    quiet_threshold_rms = 0.2 * average_volume

    # Align lengths safely
    L = min(len(drum_rms), len(bass_rms), len(other_rms), len(vocals_rms), len(rms))

    drum_quiet = (drum_rms[:L] < quiet_threshold_drums)
    bass_quiet = (bass_rms[:L] < quiet_threshold_bass)
    other_quiet = (other_rms[:L] < quiet_threshold_other)
    vocals_quiet = (vocals_rms[:L] < quiet_threshold_vocals)
    rms_quiet = (rms[:L] < quiet_threshold_rms)

    combined_quiet = drum_quiet.astype(np.int32) + bass_quiet.astype(np.int32) + other_quiet.astype(np.int32) + vocals_quiet.astype(np.int32) + rms_quiet.astype(np.int32)

    window_size = 22
    min_quiet_frames = 20

    # Window counts where at least 3 tracks are quiet
    quiet_mask = (combined_quiet >= 3).astype(np.int32)
    counts = np.convolve(quiet_mask, np.ones(window_size, dtype=np.int32), mode='valid')
    window_ok = (counts >= min_quiet_frames).astype(np.int32)

    # Expand window flags to full-length coverage
    coverage = np.convolve(window_ok, np.ones(window_size, dtype=np.int32), mode='full')[:L] > 0

    # Extract ranges
    padded = np.pad(coverage.astype(np.int8), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    # Cast to Python ints for JSON safety
    quiet_ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]

    silent_pre_segments = []
    fps = 43
    for section in quiet_ranges:
        for segment in struct_data["segments"]:
            if abs(int(segment["start"] * fps) - section[1]) < 100:
                silent_pre_segments.append((int(section[0]), int(section[1])))
                break

    return silent_pre_segments, quiet_ranges

def get_pauses_for_segment(rms, threshold):
    rms = np.asarray(rms, dtype=np.float64)
    quiet = rms < threshold

    window_size = 50
    min_quiet_frames = 40

    # Counts of quiet frames per window
    counts = np.convolve(quiet.astype(np.int32), np.ones(window_size, dtype=np.int32), mode='valid')
    window_ok = (counts >= min_quiet_frames).astype(np.int32)

    # Expand window flags to full-length coverage
    coverage = np.convolve(window_ok, np.ones(window_size, dtype=np.int32), mode='full')[:len(rms)] > 0

    # Extract ranges
    padded = np.pad(coverage.astype(np.int8), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    # Cast to Python ints for JSON safety
    quiet_ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return quiet_ranges
