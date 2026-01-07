import librosa
import numpy as np
from src.services.audio_analysis import drum_analysis_strobes
from src.services.audio_analysis import drum_analysis_dimmer
from src.services.audio_analysis.seperate_drums_larsnet import separate_drums_with_larsnet


def initialize_song_metrics(song_data, struct_data=None):
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

    enriched_segments = drum_analysis_dimmer.analyze_drum_beat_pattern(
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

    if rms == False:
        rms = struct_data['rms']

    drums_rms = struct_data['drums_rms']
    bass_rms = struct_data['bass_rms']
    drums_rms = [drums_rms[i] + bass_rms[i] for i in range(len(drums_rms))]

    segments = struct_data['segments']
    song_rms = struct_data['rms']

    total_volumes = []
    avg_volumes = []
    drum_volumes = []
    loud_sections = []
    focus = {}
    is_quiet = False
    labels = {}
    i = 0
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

        rms_slice = rms[int(segment_start*43):int(segment_end*43)]
        temp_avg = sum(rms_slice) / len(rms_slice)
        pauses = get_pauses_for_segment(rms_slice, temp_avg*0.2)
        rms_slice = get_modified_rms(song_data, name=name, rms=rms_slice, pauses=pauses)
        rms_slice = [i for i in rms_slice if i != 0]

        drums_slice = drums_rms[int(segment_start*43):int(segment_end*43)]
        temp_avg = sum(drums_slice) / len(drums_slice)
        pauses = get_pauses_for_segment(drums_slice, temp_avg*0.2)
        drums_slice = get_modified_rms(song_data, name=name, rms=drums_slice, pauses=pauses)
        drums_slice = [i for i in drums_slice if i != 0]

        song_slice = song_rms[int(segment_start*43):int(segment_end*43)]
        temp_avg = sum(song_slice) / len(song_slice)
        pauses = get_pauses_for_segment(song_slice, temp_avg*0.2)
        song_slice = get_modified_rms(song_data, name=name, rms=song_slice, pauses=pauses)
        song_slice = [i for i in song_slice if i != 0]

        if len(rms_slice) > 0:
            average_rms = sum(rms_slice) / len(rms_slice)
        else:
            average_rms = 0    
        if len(drums_slice) > 0:
            average_drums = sum(drums_slice) / len(drums_slice)
        else:
            average_drums = 0
        if len(song_slice) > 0:
            average_total_rms = sum(song_slice) / len(song_slice)
        else:
            average_total_rms = 0
        segment["avg_combined"] = average_total_rms
        segment['avg_volume'] = average_rms
        segment['avg_drums'] = average_drums
        avg_volumes.append(average_rms)
        total_volumes.append(average_total_rms)
        drum_volumes.append(average_drums)
        i += 1
    song_average = sum(avg_volumes) / len(avg_volumes)
    total_rms = sum(song_rms) / len(song_rms)
    drum_average = sum(drum_volumes) / len(drum_volumes)

    verses = [segment for segment in segments if segment['label'] == 'verse']
    if len(verses) != 0:
        verses_avg = sum([segment['avg_combined'] for segment in verses]) / len(verses)
        verses_avg2 = sum([segment['avg_volume'] for segment in verses]) / len(verses)
    else:
        verses_avg = 0
        verses_avg2 = 0
    
    choruses = [segment for segment in segments if segment['label'] == 'chorus']
    if len(choruses) != 0:
        choruses_avg = sum([segment['avg_combined'] for segment in choruses]) / len(choruses)
        choruses_avg2 = sum([segment['avg_volume'] for segment in choruses]) / len(choruses)
    else:
        choruses_avg = 0
        choruses_avg2 = 0

    if len(verses) != 0:
        versesdrum_average = sum([segment['avg_drums'] for segment in verses]) / len(verses)
    if len(choruses) != 0:
        chorusesdrum_average = sum([segment['avg_drums'] for segment in choruses]) / len(choruses)
    else:
        chorusesdrum_average = 0

    if "inst" in labels:
        inst = [segment for segment in segments if segment['label'] == 'inst']
        inst_average = sum([segment['avg_combined'] for segment in inst]) / len(inst)
        inst_average2 = sum([segment['avg_volume'] for segment in inst]) / len(inst)
        average_volumes = {"verse": (verses_avg, verses_avg2), "chorus": (choruses_avg, choruses_avg2), "inst": (inst_average, inst_average2) }
    else:
        average_volumes = {"verse": (verses_avg, verses_avg2), "chorus": (choruses_avg, choruses_avg2)}
    if choruses_avg > total_rms:
        loud_sections.append(("chorus", (choruses_avg, choruses_avg2)))
    if verses_avg > total_rms:
        loud_sections.append(("verse", (verses_avg, verses_avg2)))
    if "inst" in labels:
        if inst_average > total_rms:
            loud_sections.append(("inst", (inst_average, inst_average2)))
    loud_sections_average = sum([section[1][0] for section in loud_sections]) / len(loud_sections)
    loud_sections_average2 = sum([section[1][1] for section in loud_sections]) / len(loud_sections)
    # Sort sections by average loudness in descending order
    sorted_sections = sorted(loud_sections, key=lambda item: item[1][0], reverse=True)
    # Assign sections to focus dictionary based on loudness
    focuses = ["first", "second", "third"]
    for i in range(len(sorted_sections)):
        focus[focuses[i]] = sorted_sections[i][0]

    params.append({'segments': segments, 'average_rms': song_average,
                    "loud_sections": [section[0] for section in loud_sections], 
                    "focus": focus, "is_quiet": is_quiet, "loud_sections_average": loud_sections_average,
                      "loud_sections_average2": loud_sections_average2, "average_volumes": average_volumes,
                        "drum_average": drum_average, "broken_bridges": broken_bridges})
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
    if type(rms) != list:
        rms, average = get_rms(song_data, category=category)
    else:
        rms = rms
    if type(pauses) != list:
        pauses = get_pauses(name, struct_data)
    else:
        pauses = pauses

    # Initialize the modified RMS list
    modified_rms = rms

    # Set the RMS values between the pauses to 0
    for pause in pauses:
        start, end = pause[0], pause[1]
        for i in range(start, end):
            modified_rms[i] = 0

    return modified_rms

def get_pauses(name, struct_data):
    # Calculate the RMS of the paths
    drum_rms = struct_data["drums_rms"]
    bass_rms = struct_data["bass_rms"]
    other_rms = struct_data["other_rms"]
    vocals_rms = struct_data["vocals_rms"]
    rms = struct_data["rms"]
    
    average_volume = struct_data["total_rms"]
    # Define a threshold for what constitutes a "quiet" section
    quiet_threshold_drums = 0.2*struct_data["drums_average"]
    quiet_threshold_bass = 0.2*struct_data["bass_average"]
    quiet_threshold_other = 0.2*struct_data["other_average"]
    quiet_threshold_vocals = 0.2*struct_data["vocals_average"]
    quiet_threshold_rms = 0.2*average_volume
    
    # Find the quiet sections in each track
    drum_quiet = [x < quiet_threshold_drums for x in drum_rms]
    bass_quiet = [x < quiet_threshold_bass for x in bass_rms]
    other_quiet = [x < quiet_threshold_other for x in other_rms]
    vocals_quiet = [x < quiet_threshold_vocals for x in vocals_rms]
    rms_quiet = [x < quiet_threshold_rms for x in rms]

    combined_quiet = [0] * max(len(drum_quiet), len(bass_quiet), len(other_quiet), len(vocals_quiet))
    # Combine the quiet sections into a single array
    for i in range(max(len(drum_quiet), len(bass_quiet), len(other_quiet), len(vocals_quiet), len(rms_quiet))):
        if drum_quiet[i]:
            combined_quiet[i] += 1
        if bass_quiet[i]:
            combined_quiet[i] += 1
        if other_quiet[i]:
            combined_quiet[i] += 1
        if vocals_quiet[i]:
            combined_quiet[i] += 1
        if rms_quiet[i]:
            combined_quiet[i] += 1
    # Define the window size and the minimum number of quiet frames
    window_size = 22
    min_quiet_frames = 20

    # Initialize an array to hold the quiet sections
    quiet_sections = [0] * len(combined_quiet)

    # For each window of frames
    for i in range(len(combined_quiet) - window_size + 1):
        # Count how many frames in the window are quiet
        quiet_count = sum(1 for j in range(i, i + window_size) if combined_quiet[j] >= 3)
        
        # If at least min_quiet_frames are quiet, mark the entire window as a quiet section
        if quiet_count >= min_quiet_frames:
            for j in range(i, i + window_size):
                quiet_sections[j] = 1

    # Find the beginning and end indexes of the quiet sections
    quiet_ranges = []
    start_index = None
    for i, is_quiet in enumerate(quiet_sections):
        if is_quiet and start_index is None:
            start_index = i
        elif not is_quiet and start_index is not None:
            quiet_ranges.append((start_index, i))
            start_index = None

    # If the last section is quiet, add it to the list
    if start_index is not None:
        quiet_ranges.append((start_index, len(quiet_sections)))
    silent_pre_segments = []
    for section in quiet_ranges:
        for segment in struct_data["segments"]:
            if abs(segment["start"]*43 - section[1]) < 100: 
                silent_pre_segments.append(section)
                break
    return silent_pre_segments, quiet_ranges

def get_pauses_for_segment(rms, threshold):
    # Define a threshold for what constitutes a "quiet" section
    quiet_threshold = threshold
    # Find the quiet sections in each track
    quiet = [x < quiet_threshold for x in rms]
    # Define the window size and the minimum number of quiet frames
    window_size = 50
    min_quiet_frames = 40
    # Initialize an array to hold the quiet sections
    quiet_sections = [0] * len(quiet)
    # For each window of frames
    for i in range(len(quiet) - window_size + 1):
        # Count how many frames in the window are quiet
        quiet_count = sum(1 for j in range(i, i + window_size) if quiet[j])
        # If at least min_quiet_frames are quiet, mark the entire window as a quiet section
        if quiet_count >= min_quiet_frames:
            for j in range(i, i + window_size):
                quiet_sections[j] = 1
    # Find the beginning and end indexes of the quiet sections
    quiet_ranges = []
    start_index = None
    for i, is_quiet in enumerate(quiet_sections):
        if is_quiet and start_index is None:
            start_index = i
        elif not is_quiet and start_index is not None:
            quiet_ranges.append((start_index, i))
            start_index = None
    # If the last section is quiet, add it to the list
    if start_index is not None:
        quiet_ranges.append((start_index, len(quiet_sections)))
    return quiet_ranges
