import librosa
import numpy as np

def detect_onsets(y, sr, threshold):
    # Compute the multi-channel onset strength
    onset_env_multi = librosa.onset.onset_strength_multi(y=y, sr=sr)

    # Initialize an empty list to store the times of the onsets
    onsets = []

    # Iterate over each frequency band
    for i, onset_env in enumerate(onset_env_multi):
        # Normalize the onset strength
        onset_env = librosa.util.normalize(onset_env)

        # Find the times where the onset strength exceeds the threshold
        times = np.where(onset_env > threshold)[0]

        # Convert the times from frames to seconds
        times = librosa.frames_to_time(times, sr=sr)

        # Add the times to the list of onsets
        onsets.append(times)

    return onsets

def onset_rate_per_segment(onsets, segments):
    # Initialize an empty dictionary to store the onset rate for each segment
    onset_rates = {}

    # Iterate over the segments
    for i, segment in enumerate(segments):
        # Filter the onsets that fall within the start and end times of the segment
        segment_onsets = [time for times in onsets for time in times if segment['start'] <= time <= segment['end']]

        # Calculate the duration of the segment
        duration = segment['end'] - segment['start']

        # Calculate the onset rate for the segment
        onset_rates[i] = len(segment_onsets) / duration if duration > 0 else 0

    return onset_rates

def merge_close_values(values, threshold):
    # Sort the values
    values.sort()

    # Initialize the list of merged values with the first value
    merged = [values[0]]

    # Iterate over the rest of the values
    for value in values[1:]:
        # If the current value is close to the last merged value, replace the last merged value with their average
        if value - merged[-1] <= threshold:
            merged[-1] = (merged[-1] + value) / 2
        else:
            # Otherwise, add the current value to the list of merged values
            merged.append(value)

    return merged

def group_close_values(values, segments, rates):
    # Sort the values
    values.sort()

    # Initialize the list of groups with the first value
    groups = [[values[0]]]

    # Iterate over the rest of the values
    for value in values[1:]:
        for i, segment in enumerate(segments):
            if segment['start'] <= value <= segment['end']:
                threshold = (1/rates[i])
                if threshold > 0.25:
                    threshold = 0.25
                if threshold < 0.13:
                    threshold = 0.13
        # If the current value is close to the last value in the last group, add it to that group
        if value - groups[-1][-1] <= threshold:
            groups[-1].append(value)
        else:
            # Only add the last group to the list if it has more than two entries
            if len(groups[-1]) > 2:
                groups.append(groups[-1])
            # Start a new group with the current value
            groups[-1] = [value]

    # Add the last group to the list if it has more than two entries
    if len(groups[-1]) > 2:
        groups.append(groups[-1])

    return groups

def fuse_close_groups(grouped_onsets):
    fused_groups = []
    current_group = grouped_onsets[0]

    for next_group in grouped_onsets[1:]:
        # If the start of the next group is within 0.35s of the end of the current group
        if next_group[0] - current_group[-1] <= 0.35:
            # Extend the current group with the next group
            current_group += (next_group)
        else:
            # Add the current group to the list of fused groups
            fused_groups.append(current_group)
            # Start a new current group with the next group
            current_group = next_group

    # Add the last current group to the list of fused groups
    fused_groups.append(current_group)

    return fused_groups
            
def convert_to_start_end_times(grouped_onsets):
    start_end_times = []
    for group in grouped_onsets:
        if group:  # if the group is not empty
            start = group[0]
            end = group[-1]
            start_end_times.append((start, end))
    return start_end_times

def get_onset_parts(segments=None, input=None, sr=None):
    if input is None or sr is None:
        return []
    if segments is None:
        duration = librosa.get_duration(y=input, sr=sr)
        segments = [{"start": 0, "end": duration}]

    onsets = detect_onsets(input, sr, threshold=0.15)
    merged_onsets = [merge_close_values(times, threshold=0.055) for times in onsets]

    segment_rates = onset_rate_per_segment(merged_onsets, segments)

    grouped_onsets = [group_close_values(times, segments, segment_rates) for times in merged_onsets]
    grouped_onsets = grouped_onsets[0]

    fused_onsets = fuse_close_groups(grouped_onsets)
    fused_onsets = convert_to_start_end_times(fused_onsets)
    return fused_onsets

# Example usage
if __name__ == "__main__":
    your_filepath = "/home/tumffa/aidmx/demix/htdemucs/mycurse/drums.wav" # full path to your file
    # Load the audio file
    y, sr = librosa.load(your_filepath, sr=None, mono=True)
    segments = None # set this to [ {"start": 0, "end": 10}, {"start": 10, "end": 20} ] etc. if you want specific times
    onsets = get_onset_parts(segments=segments, input=y, sr=sr)
    print(onsets)