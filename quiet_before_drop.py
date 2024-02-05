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
    
    struct_data = DataService.get_struct_data(name)
    average_volume = struct_data["total_rms"]
    # Define a threshold for what constitutes a "quiet" section
    quiet_threshold_drums = 0.15*average_volume
    quiet_threshold_bass = 0.15*average_volume
    quiet_threshold_other = 0.15*average_volume
    quiet_threshold_vocals = 0.15*average_volume

    # Find the quiet sections in each track
    drum_quiet = [x < quiet_threshold_drums for x in drum_rms]
    bass_quiet = [x < quiet_threshold_bass for x in bass_rms]
    other_quiet = [x < quiet_threshold_other for x in other_rms]
    vocals_quiet = [x < quiet_threshold_vocals for x in vocals_rms]

    combined_quiet = [0] * max(len(drum_quiet), len(bass_quiet), len(other_quiet), len(vocals_quiet))
    # Combine the quiet sections into a single array
    for i in range(max(len(drum_quiet), len(bass_quiet), len(other_quiet), len(vocals_quiet))):
        if drum_quiet[i]:
            combined_quiet[i] += 1
        if bass_quiet[i]:
            combined_quiet[i] += 1
        if other_quiet[i]:
            combined_quiet[i] += 1
        if vocals_quiet[i]:
            combined_quiet[i] += 1
    # Define the window size and the minimum number of quiet frames
    window_size = 25
    min_quiet_frames = 20

    # Initialize an array to hold the quiet sections
    quiet_sections = [0] * len(combined_quiet)

    # For each window of frames
    for i in range(len(combined_quiet) - window_size + 1):
        # Count how many frames in the window are quiet
        quiet_count = sum(1 for j in range(i, i + window_size) if combined_quiet[j] >= 2)
        
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

    # for start_index, end_index in quiet_ranges:
    #     print('Quiet section from {} to {}'.format(start_index, end_index))

    data = DataService.get_struct_data(name)
    segments = data["segments"]

    # Sort the quiet_ranges in descending order of end time
    quiet_ranges.sort(key=lambda x: x[1], reverse=True)
    sr = 43
    # Initialize a list to hold the silent parts that precede a segment change
    silent_pre_segments = []

    # Initialize a variable to hold the last quiet range added
    last_quiet_range = None
    # For each segment
    for segment in segments:
        # Convert the segment start time to frames
        segment_start_frame = int(segment['start'] * sr)
        #1858 to 1975
        # Find the preceding quiet range
        for i in range(len(quiet_ranges)):
            # Calculate the difference in frames between the segment start and the end of the quiet range
            diff_frames1 = abs(segment_start_frame - quiet_ranges[i][1])
            diff_frames2 = abs(segment_start_frame - quiet_ranges[i][0])
            # print(f"Comparing {segment_start_frame} to {quiet_ranges[i]} with diff {diff_frames1} and {diff_frames2} with volume {drum_rms[0][i]}")
            
            # Convert 2.5 seconds to frames
            diff_frames_threshold = int(1.5 * sr)
            
            # If the quiet range ends before the segment starts and it's not the same as the last one added
            # and the difference in frames is less than or equal to the threshold
            if (last_quiet_range is None or quiet_ranges[i] != last_quiet_range) and (diff_frames1 <= diff_frames_threshold or diff_frames2 <= diff_frames_threshold):
                # Add it to the list and update last_quiet_range
                silent_pre_segments.append(quiet_ranges[i])
                last_quiet_range = quiet_ranges[i]
                break
            
    return silent_pre_segments

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