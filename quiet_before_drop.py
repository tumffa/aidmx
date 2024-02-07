import librosa
import json
import StatisticsService
import DataService

def get_pauses(name, data):
    print(f"Getting pauses for {name}")

    struct = DataService.get_struct_data(name)
    # Calculate the RMS of the paths
    drum_rms = struct["drums_rms"]
    bass_rms = struct["bass_rms"]
    other_rms = struct["other_rms"]
    vocals_rms = struct["vocals_rms"]
    rms = struct["rms"]
    
    struct_data = DataService.get_struct_data(name)
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
        for segment in struct["segments"]:
            if abs(segment["start"]*43 - section[1]) < 100: 
                silent_pre_segments.append(section)
                break
    print("------------------PAUSES--------------------")        
    print(silent_pre_segments)
    print("------------------PAUSE RANGES--------------------")
    print(quiet_ranges)
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
        print(quiet_ranges)
    return quiet_ranges