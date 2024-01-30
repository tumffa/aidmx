import librosa
import json
from StatisticsService import get_rms

def get_pauses(name, data):
    drums_path = f"{data['demixed']}/drums.wav"
    bass_path = f"{data['demixed']}/bass.wav"
    other_path = f"{data['demixed']}/other.wav"
    vocals_path = f"{data['demixed']}/vocals.wav"

    # Calculate the RMS of the paths
    drum_rms = get_rms(path=drums_path)
    bass_rms = get_rms(path=bass_path)
    other_rms = get_rms(path=other_path)
    vocals_rms = get_rms(path=vocals_path)
    
    # Define a threshold for what constitutes a "quiet" section
    quiet_threshold = 0.05

    # Find the quiet sections in each track
    drum_quiet = drum_rms < quiet_threshold
    bass_quiet = bass_rms < quiet_threshold
    other_quiet = other_rms < quiet_threshold
    vocals_quiet = vocals_rms < quiet_threshold

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
    window_size = 30
    min_quiet_frames = 22

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

    for start_index, end_index in quiet_ranges:
        print('Quiet section from {} to {}'.format(start_index, end_index))

    with open(f"./struct/{name}vocals.json", 'r') as f:
        data = json.load(f)
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
            print(f"Comparing {segment_start_frame} to {quiet_ranges[i]} with diff {diff_frames1} and {diff_frames2}")
            
            # Convert 2.5 seconds to frames
            diff_frames_threshold = int(1.5 * sr)
            
            # If the quiet range ends before the segment starts and it's not the same as the last one added
            # and the difference in frames is less than or equal to the threshold
            if (last_quiet_range is None or quiet_ranges[i] != last_quiet_range) and (diff_frames1 <= diff_frames_threshold or diff_frames2 <= diff_frames_threshold):
                # Add it to the list and update last_quiet_range
                silent_pre_segments.append(quiet_ranges[i])
                last_quiet_range = quiet_ranges[i]
                break

    print(silent_pre_segments)