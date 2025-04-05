import os
import librosa
import numpy as np
from scipy import signal
    
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
    
def analyze_drum_patterns(demix_path, beats=None, segments=None, sr=22050, hop_length=512):
    """
    Use autocorrelation to find rhythmic patterns in kick and snare hits for each segment.
    
    Args:
        demix_path: Path to the directory containing separated drum tracks
        segments: List of segment dictionaries with 'start', 'end', and 'label' keys
        beats: List of beat times (in seconds)
        sr: Sample rate
        hop_length: Hop length used in RMS calculation
        
    Returns:
        The segments list with added pattern data for each segment
    """

    # Define file paths for kick and snare components
    kick_path = os.path.join(demix_path, "drums", "kick", "drums.wav")
    snare_path = os.path.join(demix_path, "drums", "snare", "drums.wav")
    toms_path = os.path.join(demix_path, "drums", "toms", "drums.wav")
    
    print(f"Loading drum tracks from: {demix_path}")
    print(f"Kick path: {kick_path}")
    print(f"Snare path: {snare_path}")
    
    # Load audio files
    try:
        kick_audio, sr_kick = librosa.load(kick_path, sr=sr, mono=True)
        print(f"Found kick at {kick_path}")
        # Average kick audio volume
        kick_audio_avg = np.mean(np.abs(kick_audio))
    except Exception as e:
        print(f"Error loading kick track: {e}")
        kick_audio = np.zeros(1000)  # Fallback empty audio
    
    try:
        snare_audio, sr_snare = librosa.load(snare_path, sr=sr, mono=True)
        print(f"Found snare at {snare_path}")
        # Average snare audio volume
        snare_audio_avg = np.mean(np.abs(snare_audio))
    except Exception as e:
        print(f"Error loading snare track: {e}")
        snare_audio = np.zeros(1000)  # Fallback empty audio

    #normalize audio
    if np.max(kick_audio) > 0:
        kick_audio = kick_audio / np.max(kick_audio)
    if np.max(snare_audio) > 0:
        snare_audio = snare_audio / np.max(snare_audio)
    
    # Calculate onset strength for each component
    kick_onset_env = librosa.onset.onset_strength(y=kick_audio, sr=sr, hop_length=hop_length)
    snare_onset_env = librosa.onset.onset_strength(y=snare_audio, sr=sr, hop_length=hop_length)
    
    # IMPORTANT: Normalize the onset envelopes globally (not per segment)
    # This preserves the relative strength between segments
    kick_onset_env_normalized = librosa.util.normalize(kick_onset_env) if np.max(kick_onset_env) > 0 else kick_onset_env
    snare_onset_env_normalized = librosa.util.normalize(snare_onset_env) if np.max(snare_onset_env) > 0 else snare_onset_env
    
    # If no segments provided, create a single segment for the entire track
    if segments is None or len(segments) == 0:
        duration = max(
            librosa.get_duration(y=kick_audio, sr=sr),
            librosa.get_duration(y=snare_audio, sr=sr)
        )
        segments = [{"start": 0, "end": duration, "label": "entire_track"}]
    
    # Function to convert frame index to time in seconds
    def frame_to_time(frame_idx):
        return librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)
    
    # Function to convert time to frame index
    def time_to_frame(t):
        return librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
    
    def analyze_component(audio, onset_env, component_name, start_time, end_time, percentile_threshold=75):
        """Analyze a drum component using onset detection directly from audio"""
        # Convert time to frames
        start_frame = max(0, time_to_frame(start_time))
        end_frame = min(len(onset_env), time_to_frame(end_time))
        
        # Extract segment of interest
        segment_onset_env = onset_env[start_frame:end_frame]
        segment_audio = audio[start_frame*hop_length:end_frame*hop_length]
        
        if len(segment_onset_env) == 0:
            return {"error": f"No {component_name} data available in this segment"}
        
        # Check if the segment is very quiet overall compared to the track average
        segment_energy = np.mean(np.abs(segment_audio))
        track_energy = np.mean(np.abs(audio))
        ratio_to_average = segment_energy / (track_energy + 1e-10)  # Avoid division by zero

        # Peak Picking
        onset_frames = librosa.util.peak_pick(
            segment_onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.15,
            wait=2
        )
        
        # Convert to absolute frame indices
        onset_frames = onset_frames + start_frame
        
        # Create a list of (frame, strength) tuples
        onset_frames_with_strength = []
        for frame in onset_frames:
            if frame < len(onset_env):
                strength = onset_env[frame]  # Get the strength value at this frame
                onset_frames_with_strength.append((frame, float(strength)))
        
        # Convert frames to times and keep the strength values
        onset_times_with_strength = [(float(frame_to_time(frame)), strength) 
                                for frame, strength in onset_frames_with_strength]
        
        # Just the times for compatibility with existing code
        onset_times = [t for t, _ in onset_times_with_strength]
        
        print(f"  {component_name.capitalize()} - Detected {len(onset_times)} hits using onset detection")
        
        # Simplified return - periodicity will be calculated after beat matching
        return {
            "num_hits": len(onset_times),
            "hit_times": [float(t) for t in onset_times],
            "hit_times_with_strength": onset_times_with_strength,
            "pattern_found": len(onset_times) >= 4,  # Just indicate if we have sufficient hits
            "ratio_to_average": float(ratio_to_average),
        }

    # New function to calculate periodicity from beat-matched hits
    def calculate_periodicity_from_matches(matched_hits, start_time, end_time, sr=22050, hop_length=512):
        """
        Calculate periodicity information from beat-matched hits.
        
        Args:
            matched_hits: List of (hit_time, beat_time, strength) tuples
            start_time: Start time of the segment
            end_time: End time of the segment
            
        Returns:
            Dictionary with periodicity information
        """
        # Extract hit times (first element of each tuple)
        hit_times = [hit[0] for hit in matched_hits]
        
        if len(hit_times) < 4:
            return {
                "pattern_found": False,
                "period_seconds": 0,
                "period_timeframes": [],
                "tempo_bpm": 0
            }
        
        # Sort the hit times
        hit_times.sort()
        
        # Calculate inter-onset intervals (IOIs)
        iois = [hit_times[i+1] - hit_times[i] for i in range(len(hit_times)-1)]
        
        if not iois:
            return {
                "pattern_found": False,
                "period_seconds": 0,
                "period_timeframes": [],
                "tempo_bpm": 0
            }
            
        # Histogram approach to find the dominant period
        # Group IOIs into bins to find the most common duration
        min_period = 0.2  # 300 BPM max
        max_period = 2.0  # 30 BPM min
        
        # Filter IOIs within reasonable range
        valid_iois = [ioi for ioi in iois if min_period <= ioi <= max_period]
        
        if not valid_iois:
            # If no valid IOIs in range, use the mean of all IOIs as fallback
            period_seconds = np.mean(iois)
        else:
            # Use histogram to find most common IOI
            hist, bins = np.histogram(valid_iois, bins=20, range=(min_period, max_period))
            most_common_idx = np.argmax(hist)
            period_seconds = (bins[most_common_idx] + bins[most_common_idx+1]) / 2
        
        # Calculate BPM from period
        tempo_bpm = 60 / period_seconds
        
        # Find optimal phase alignment
        # Test different offsets to find the one that minimizes distance to hit times
        num_test_points = 50
        best_offset = 0
        min_total_distance = float('inf')
        
        for i in range(num_test_points):
            test_offset = (i / num_test_points) * period_seconds
            grid_points = []
            
            # Generate grid points for this offset
            current_time = start_time + test_offset
            while current_time <= end_time:
                grid_points.append(current_time)
                current_time += period_seconds
                
            # Calculate total distance from hits to nearest grid points
            total_distance = 0
            for hit_time in hit_times:
                nearest_distance = min(abs(hit_time - grid_point) for grid_point in grid_points)
                total_distance += nearest_distance
                
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_offset = test_offset
        
        # Generate final grid of period timeframes using best offset
        period_timeframes = []
        current_time = start_time + (best_offset % period_seconds)
        
        # Adjust first_timeframe to before start_time if needed
        while current_time > start_time:
            current_time -= period_seconds
            
        # Generate timeframes
        while current_time <= end_time:
            if current_time >= start_time:
                period_timeframes.append(float(current_time))
            current_time += period_seconds
        
        print(f"  Calculated periodicity from beat-matches: {period_seconds:.3f}s, {tempo_bpm:.1f} BPM")
        print(f"  Generated {len(period_timeframes)} phase-aligned timeframes")
        
        return {
            "pattern_found": True,
            "period_seconds": float(period_seconds),
            "tempo_bpm": float(tempo_bpm),
            "period_timeframes": period_timeframes
        }
        
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        original_label = segment.get('label', 'unnamed')
        
        print(f"\nAnalyzing segment '{original_label}' ({start_time:.2f}s - {end_time:.2f}s)")
        
        # Analyze patterns for this segment - simplified analysis without periodicity
        kick_analysis = analyze_component(kick_audio, kick_onset_env_normalized, "kick", start_time, end_time)
        snare_analysis = analyze_component(snare_audio, snare_onset_env_normalized, "snare", start_time, end_time)
        
        # Find beat snare/kick hits (without periodicity parameters since we don't have them yet)
        beat_kick_matches, beat_snare_matches = match_beats_with_drum_hits(
            beats, 
            kick_analysis.get("hit_times_with_strength", []), 
            snare_analysis.get("hit_times_with_strength", []),
            window=0.1
        )
        
        # Now calculate periodicity from the matched hits
        kick_periodicity = calculate_periodicity_from_matches(beat_kick_matches, start_time, end_time)
        snare_periodicity = calculate_periodicity_from_matches(beat_snare_matches, start_time, end_time)
        
        # Update the analysis with periodicity information and beat matches
        kick_analysis.update(kick_periodicity)
        snare_analysis.update(snare_periodicity)
        kick_analysis["beat_matches"] = beat_kick_matches
        snare_analysis["beat_matches"] = beat_snare_matches
        
        # Overall tempo estimate (takes the more confident one)
        tempo_bpm = None
        if kick_analysis.get("pattern_found", False) and snare_analysis.get("pattern_found", False):
            # For simplicity, just take the average of kick and snare tempos
            tempo_bpm = (kick_analysis["tempo_bpm"] + snare_analysis["tempo_bpm"]) / 2
        elif kick_analysis.get("pattern_found", False):
            tempo_bpm = kick_analysis["tempo_bpm"]
        elif snare_analysis.get("pattern_found", False):
            tempo_bpm = snare_analysis["tempo_bpm"]
        
        # Add the results to this segment
        segment["drum_analysis"] = {
            "kick": kick_analysis,
            "snare": snare_analysis,
            "tempo_bpm": tempo_bpm,
            "n_of_kicks": kick_analysis.get("num_hits", 0),
            "n_of_snares": snare_analysis.get("num_hits", 0),
            "kick_ratio_to_average": kick_analysis.get("ratio_to_average", 0),
            "snare_ratio_to_average": snare_analysis.get("ratio_to_average", 0),
        }
        
        # Log results for this segment
        label = segment.get('label', f"{segment['start']}-{segment['end']},")
        print(f"Segment '{label}', Start: {segment['start']:.2f}, subsegment: {segment.get('is_subsegment', False)}")
        print(f"  Kick ratio to average: {kick_analysis.get('ratio_to_average', 0):.2f}")
        print(f"  Snare ratio to average: {snare_analysis.get('ratio_to_average', 0):.2f}")
        print(f"  Tempo: {tempo_bpm} BPM")
        
        if kick_analysis.get("pattern_viz"):
            print(f"  Kick pattern: {kick_analysis.get('pattern_viz')}")
        if snare_analysis.get("pattern_viz"):
            print(f"  Snare pattern: {snare_analysis.get('pattern_viz')}")
            
        print(f"  Number of kicks: {kick_analysis.get('num_hits', 0)}")
        print(f"  Number of snares: {snare_analysis.get('num_hits', 0)}")
        print(f"  Kick frames: {kick_analysis.get('hit_times_with_strength', [])}")
        print(f"  Snare frames: {snare_analysis.get('hit_times_with_strength', [])}")

    print("\n=== ANALYZING PATTERNS ACROSS SEGMENT TYPES ===")
    summary_results = analyze_drum_patterns_by_label(segments)
    
    # Add summary to each segment for reference
    for segment in segments:
        label = segment.get('label', 'unknown')
        if label in summary_results["summary_by_label"]:
            segment["drum_analysis"]["label_summary"] = summary_results["summary_by_label"][label]
    
    return segments

def analyze_drum_patterns_by_label(segments):
    """
    Analyze and summarize drum pattern metrics across segments with the same label.
    
    Args:
        segments: List of segment dictionaries with drum_analysis already computed
        
    Returns:
        Dictionary containing summary metrics by label and all segment data
    """
    # Group segments by label
    label_groups = {}
    for segment in segments:
        label = segment.get('label', 'unknown')
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(segment)
    
    print("\n=== DRUM PATTERN ANALYSIS BY SEGMENT TYPE ===")
    
    # Initialize results dictionary
    results = {
        "summary_by_label": {},
        "all_segments": segments
    }
    
    # Process each label group
    for label, segment_group in label_groups.items():
        print(f"\n--- {label.upper()} SEGMENTS ({len(segment_group)} instances) ---")
        
        # Collect metrics
        kick_tempos = []
        snare_tempos = []
        overall_tempos = []
        kick_periods = []
        snare_periods = []
        kick_hits_per_seg = []
        snare_hits_per_seg = []
        kick_strength = []
        snare_strength = []
        kick_patterns = []
        snare_patterns = []
        segment_durations = []
        
        # Extract metrics from each segment
        for segment in segment_group:
            if 'drum_analysis' not in segment:
                continue
                
            drum_analysis = segment['drum_analysis']
            segment_duration = segment['end'] - segment['start']
            segment_durations.append(segment_duration)
            
            # Get kick metrics
            if 'kick' in drum_analysis and drum_analysis['kick'].get('pattern_found', False):
                kick_tempos.append(drum_analysis['kick'].get('tempo_bpm', 0))
                kick_periods.append(drum_analysis['kick'].get('period_seconds', 0))
                kick_hits_per_seg.append(drum_analysis['n_of_kicks'])
                kick_strength.append(drum_analysis['kick'].get('pattern_strength', 0))
                kick_patterns.append(drum_analysis['kick'].get('binary_pattern', ''))
            
            # Get snare metrics
            if 'snare' in drum_analysis and drum_analysis['snare'].get('pattern_found', False):
                snare_tempos.append(drum_analysis['snare'].get('tempo_bpm', 0))
                snare_periods.append(drum_analysis['snare'].get('period_seconds', 0))
                snare_hits_per_seg.append(drum_analysis['n_of_snares'])
                snare_strength.append(drum_analysis['snare'].get('pattern_strength', 0))
                snare_patterns.append(drum_analysis['snare'].get('binary_pattern', ''))
            
            # Overall tempo
            if drum_analysis.get('tempo_bpm'):
                overall_tempos.append(drum_analysis['tempo_bpm'])
        
        # Compute averages (with error handling for empty lists)
        avg_overall_tempo = sum(overall_tempos) / len(overall_tempos) if overall_tempos else 0
        avg_kick_tempo = sum(kick_tempos) / len(kick_tempos) if kick_tempos else 0
        avg_snare_tempo = sum(snare_tempos) / len(snare_tempos) if snare_tempos else 0
        avg_kick_period = sum(kick_periods) / len(kick_periods) if kick_periods else 0
        avg_snare_period = sum(snare_periods) / len(snare_periods) if snare_periods else 0
        
        # Normalize hit counts by duration to get density
        kick_densities = [hits/duration for hits, duration in zip(kick_hits_per_seg, segment_durations)] if kick_hits_per_seg else []
        snare_densities = [hits/duration for hits, duration in zip(snare_hits_per_seg, segment_durations)] if snare_hits_per_seg else []
        avg_kick_density = sum(kick_densities) / len(kick_densities) if kick_densities else 0
        avg_snare_density = sum(snare_densities) / len(snare_densities) if snare_densities else 0
        
        # Find the most common pattern
        def most_common(patterns):
            if not patterns:
                return "No pattern"
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            return max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        common_kick_pattern = most_common(kick_patterns)
        common_snare_pattern = most_common(snare_patterns)
        
        # Find tempo agreement - check if there's consistency
        tempo_agreement = "high" if len(set([round(t) for t in overall_tempos])) <= 1 else "medium" if len(set([round(t/5) * 5 for t in overall_tempos])) <= 2 else "low"
        
        # Find the most representative tempo using harmonic relationships
        def find_representative_tempo(tempos):
            if not tempos:
                return 0
                
            # Group tempos into harmonic families (tempos related by factors of 2)
            tempo_families = []
            
            for tempo in sorted(tempos):
                added_to_family = False
                
                for family in tempo_families:
                    # Check if this tempo is harmonically related to the family base tempo
                    base_tempo = family[0]
                    ratio = max(tempo, base_tempo) / min(tempo, base_tempo)
                    
                    # If close to a power of 2 or common ratio
                    if abs(ratio - 1) < 0.1 or abs(ratio - 2) < 0.2 or abs(ratio - 0.5) < 0.1:
                        family.append(tempo)
                        added_to_family = True
                        break
                
                if not added_to_family:
                    tempo_families.append([tempo])
            
            # Find the largest family
            largest_family = max(tempo_families, key=len)
            
            # Return the median of the largest family
            largest_family.sort()
            if len(largest_family) % 2 == 0:
                return (largest_family[len(largest_family)//2 - 1] + largest_family[len(largest_family)//2]) / 2
            else:
                return largest_family[len(largest_family)//2]
        
        representative_tempo = find_representative_tempo(overall_tempos)
        
        # Store summary metrics
        summary = {
            "num_segments": len(segment_group),
            "avg_duration": sum(segment_durations) / len(segment_durations) if segment_durations else 0,
            "representative_tempo": representative_tempo,
            "tempo_agreement": tempo_agreement,
            "avg_overall_tempo": avg_overall_tempo,
            "avg_kick_tempo": avg_kick_tempo, 
            "avg_snare_tempo": avg_snare_tempo,
            "avg_kick_period": avg_kick_period,
            "avg_snare_period": avg_snare_period,
            "avg_kick_hits_per_segment": sum(kick_hits_per_seg) / len(kick_hits_per_seg) if kick_hits_per_seg else 0,
            "avg_snare_hits_per_segment": sum(snare_hits_per_seg) / len(snare_hits_per_seg) if snare_hits_per_seg else 0,
            "avg_kick_density": avg_kick_density,
            "avg_snare_density": avg_snare_density,
            "common_kick_pattern": common_kick_pattern,
            "common_snare_pattern": common_snare_pattern,
            "kick_pattern_consistency": len(set(kick_patterns)) == 1 if kick_patterns else False,
            "snare_pattern_consistency": len(set(snare_patterns)) == 1 if snare_patterns else False,
        }
        
        results["summary_by_label"][label] = summary
        
        # Print summary
        print(f"  Representative tempo: {representative_tempo:.1f} BPM (agreement: {tempo_agreement})")
        print(f"  Avg. segment duration: {summary['avg_duration']:.2f} seconds")
        print(f"  Kick metrics:")
        print(f"    - Avg tempo: {avg_kick_tempo:.1f} BPM")
        print(f"    - Avg period: {avg_kick_period:.3f} seconds")
        print(f"    - Avg density: {avg_kick_density:.2f} hits/second")
        print(f"    - Most common pattern: {common_kick_pattern}")
        print(f"  Snare metrics:")
        print(f"    - Avg tempo: {avg_snare_tempo:.1f} BPM") 
        print(f"    - Avg period: {avg_snare_period:.3f} seconds")
        print(f"    - Avg density: {avg_snare_density:.2f} hits/second")
        print(f"    - Most common pattern: {common_snare_pattern}")
        
        # Print any significant variations
        if tempo_agreement == "low":
            print("  Note: High tempo variation detected across segments")
            print(f"    - Tempo range: {min(overall_tempos):.1f} - {max(overall_tempos):.1f} BPM")
        
        # Print consistency info
        if len(segment_group) > 1:
            if not summary["kick_pattern_consistency"]:
                print("  Note: Kick patterns vary across segments")
            if not summary["snare_pattern_consistency"]:
                print("  Note: Snare patterns vary across segments")
                
    return results

def match_beats_with_drum_hits(beats, kick_hits, snare_hits, window=0.15, 
                              kick_period=None, snare_period=None, kick_pattern_grid=None, snare_pattern_grid=None):
    """
    Match beats with drum hits based on proximity and hit strength.
    
    Args:
        beats: List of beat timeframes (in seconds)
        kick_hits: List of (time, strength) tuples for kick hits
        snare_hits: List of (time, strength) tuples for snare hits
        window: Time window (in seconds) to consider a beat matching with a hit
        kick_period: Not used but kept for compatibility
        snare_period: Not used but kept for compatibility
        kick_pattern_grid: Not used but kept for compatibility
        snare_pattern_grid: Not used but kept for compatibility
    
    Returns:
        A tuple containing:
        - List of (kick_time, beat_time, kick_strength) for kicks with a nearby beat
        - List of (snare_time, beat_time, snare_strength) for snares with a nearby beat
    """
    # Ensure beats list is not empty to avoid errors
    if not beats:
        return [], []
    
    beat_kick_matches = []
    beat_snare_matches = []
    
    # Function to calculate beat proximity score
    def calculate_proximity_score(hit_time, beat_time, window):
        distance = abs(hit_time - beat_time)
        if distance > window:
            return 0.0
            
        # Convert to a score between 0 and 1
        return 1.0 - (distance / window)
    
    # Get maximum strengths for normalization
    max_kick_strength = max([s for _, s in kick_hits]) if kick_hits else 1.0
    max_snare_strength = max([s for _, s in snare_hits]) if snare_hits else 1.0
    
    # Process kicks
    for beat_time in beats:
        best_kick_match = None
        best_kick_score = 0.0
        
        for kick_time, kick_strength in kick_hits:
            # Calculate normalized strength score
            strength_score = kick_strength / max_kick_strength
            
            # Calculate beat proximity score
            proximity_score = calculate_proximity_score(kick_time, beat_time, window)
            if proximity_score == 0:
                continue  # Skip if not within window
            
            # Calculate combined score (weighted average)
            # Now only using proximity (80%) and strength (20%)
            combined_score = (
                (1.0 * proximity_score) +
                (0.0 * strength_score)
            )
            
            # Keep the best match
            if combined_score > best_kick_score:
                best_kick_score = combined_score
                best_kick_match = (kick_time, beat_time, kick_strength)
        
        # Add the match if the score exceeds threshold
        if best_kick_match and best_kick_score >= 0.5:
            beat_kick_matches.append(best_kick_match)
    
    # Process snares (same approach)
    for beat_time in beats:
        best_snare_match = None
        best_snare_score = 0.0
        
        for snare_time, snare_strength in snare_hits:
            # Calculate normalized strength score
            strength_score = snare_strength / max_snare_strength
            
            # Calculate beat proximity score
            proximity_score = calculate_proximity_score(snare_time, beat_time, window)
            if proximity_score == 0:
                continue  # Skip if not within window
            
            # Calculate combined score (weighted average)
            combined_score = (
                (1 * proximity_score) +
                (0.0 * strength_score)
            )
            
            # Keep the best match
            if combined_score > best_snare_score:
                best_snare_score = combined_score
                best_snare_match = (snare_time, beat_time, snare_strength)
        
        # Add the match if the score exceeds threshold
        if best_snare_match and best_snare_score >= 0.5:  # Minimum 60% score
            beat_snare_matches.append(best_snare_match)
    
    return beat_kick_matches, beat_snare_matches

# Example usage
if __name__ == "__main__":
    your_filepath = "/home/tumffa/aidmx/demix/htdemucs/mycurse/drums.wav" # full path to your file
    # Load the audio file
    y, sr = librosa.load(your_filepath, sr=None, mono=True)
    segments = None # set this to [ {"start": 0, "end": 10}, {"start": 10, "end": 20} ] etc. if you want specific times
    onsets = get_onset_parts(segments=segments, input=y, sr=sr)
    print(onsets)