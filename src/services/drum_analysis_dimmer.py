import os
import librosa
import numpy as np

# Function to convert frame index to time in seconds
def frame_to_time(frame_idx):
    return librosa.frames_to_time(frame_idx, sr=22050, hop_length=512)

# Function to convert time to frame index
def time_to_frame(t):
    return librosa.time_to_frames(t, sr=22050, hop_length=512)
    
def analyze_drum_patterns(demix_path, beats=None, segments=None):
    """
    Use demixed kick and snare tracks and beats to calculate an envelope that is used to scale
    dimmer intensity based on dominant hits. Snare is scaled higher than kick.

    Args:
        demix_path: Path to the directory containing separated drum tracks
        segments: List of segment dictionaries with 'start', 'end', and 'label' keys
        beats: List of beat times (in seconds)
    Returns:
        The segments list with added pattern data for each segment
    """
    sr = 22050
    hop_length = 512

    # Define file paths for kick and snare components
    kick_path = os.path.join(demix_path, "drums", "kick", "drums.wav")
    snare_path = os.path.join(demix_path, "drums", "snare", "drums.wav")
    
    print(f"----Calculating dimmer scaling function - loading demixed drum tracks from {demix_path}")
    try:
        kick_audio, sr_kick = librosa.load(kick_path, sr=sr, mono=True)
        print(f"------Found kick at {kick_path}")
    except Exception as e:
        print(f"------Error loading kick track: {e}")
        kick_audio = np.zeros(1000)  # Fallback empty audio
    
    try:
        snare_audio, sr_snare = librosa.load(snare_path, sr=sr, mono=True)
        print(f"------Found snare at {snare_path}")
    except Exception as e:
        print(f"------Error loading snare track: {e}")
        snare_audio = np.zeros(1000)  # Fallback empty audio

    # Normalize audio
    if np.max(kick_audio) > 0:
        kick_audio = kick_audio / np.max(kick_audio)
    if np.max(snare_audio) > 0:
        snare_audio = snare_audio / np.max(snare_audio)
    
    # Calculate onset strength for each component
    kick_onset_env = librosa.onset.onset_strength(y=kick_audio, sr=sr, hop_length=hop_length)
    snare_onset_env = librosa.onset.onset_strength(y=snare_audio, sr=sr, hop_length=hop_length)
    
    kick_onset_env_normalized = librosa.util.normalize(kick_onset_env) if np.max(kick_onset_env) > 0 else kick_onset_env
    snare_onset_env_normalized = librosa.util.normalize(snare_onset_env) if np.max(snare_onset_env) > 0 else snare_onset_env
    
    # If no segments provided, create a single segment for the entire track
    if segments is None or len(segments) == 0:
        duration = max(
            librosa.get_duration(y=kick_audio, sr=sr),
            librosa.get_duration(y=snare_audio, sr=sr)
        )
        segments = [{"start": 0, "end": duration, "label": "entire_track"}]

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        
        # Analyze patterns for this segment - simplified analysis without periodicity
        kick_analysis = analyze_component(
            kick_audio, kick_onset_env_normalized, "kick", start_time, end_time)
        snare_analysis = analyze_component(
            snare_audio, snare_onset_env_normalized, "snare", start_time, end_time)
        
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
        
        # Add the results to this segment
        segment["drum_analysis"] = {
            "kick": kick_analysis,
            "snare": snare_analysis,
            "n_of_kicks": kick_analysis.get("num_hits", 0),
            "n_of_snares": snare_analysis.get("num_hits", 0),
        }

        # Find kick/snare hits that define the beat
        find_beat_defining_hits(segment, beat_kick_matches, beat_snare_matches)

        # Calculate the light strength envelope
        light_envelope = calculate_light_strength_envelope(segment)
        
        # Add it to the segment's drum analysis
        segment["drum_analysis"]["light_strength_envelope"] = light_envelope


    print(f"----Calculated dimmer scaling function for {len(segments)} segments ({segments[0]['start']:.2f}s to {segments[-1]['end']:.2f}s)")

    return segments
    
def analyze_component(audio, onset_env, component_name, start_time, end_time):
    onset_times_with_strength = extract_component_hits(
        audio, onset_env, component_name, start_time, end_time
    )
    onset_times = [t for t, _ in onset_times_with_strength]
    
    # Simplified return - periodicity will be calculated after beat matching
    return {
        "num_hits": len(onset_times),
        "hit_times": [float(t) for t in onset_times],
        "hit_times_with_strength": onset_times_with_strength,
    }

def calculate_periodicity_from_matches(matched_hits, start_time, end_time):
    """
    Calculate periodicity information from beat-matched hits with improved phase alignment.
    
    Args:
        matched_hits: List of (hit_time, beat_time, strength) tuples
        start_time: Start time of the segment
        end_time: End time of the segment
        
    Returns:
        Dictionary with periodicity information
    """
    # Extract hit times (first element of each tuple)
    hit_times = [hit[0] for hit in matched_hits]
    # Extract associated beat times (second element of each tuple)
    beat_times = [hit[1] for hit in matched_hits]
    
    if len(hit_times) < 4:
        return {
            "period_seconds": 0,
            "period_timeframes": [],
        }
        
    # Sort the matched hits by time
    hit_times, beat_times = zip(*sorted(zip(hit_times, beat_times)))
    hit_times = list(hit_times)
    beat_times = list(beat_times)
    
    # Calculate inter-onset intervals (IOIs)
    iois = [hit_times[i+1] - hit_times[i] for i in range(len(hit_times)-1)]
    
    if not iois:
        return {
            "period_seconds": 0,
            "period_timeframes": [],
        }
        
    # Histogram approach to find the dominant period
    min_period = 0.2  # 300 BPM max
    max_period = 2.0  # 30 BPM min
    
    # Filter IOIs within reasonable range
    valid_iois = [ioi for ioi in iois if min_period <= ioi <= max_period]
    
    if not valid_iois:
        # If no valid IOIs in range, use the median of all IOIs as fallback
        period_seconds = np.median(iois) if iois else 0
    else:
        # Use histogram to find most common IOI
        hist, bins = np.histogram(valid_iois, bins=20, range=(min_period, max_period))
        most_common_idx = np.argmax(hist)
        period_seconds = (bins[most_common_idx] + bins[most_common_idx+1]) / 2
    
    # Find optimal phase by minimizing distance to actual hits
    if hit_times and period_seconds > 0:
        # Test multiple phase offsets to find the one that best aligns with the actual hits
        num_offsets = 100  # Test 100 different phase positions
        best_phase_offset = 0
        min_total_distance = float('inf')
        
        # Try different offsets throughout one period
        for i in range(num_offsets):
            test_offset = (i / num_offsets) * period_seconds
            total_distance = 0
            
            # Generate test timeframes
            test_timeframes = []
            current_time = start_time + test_offset
            while current_time <= end_time:
                test_timeframes.append(current_time)
                current_time += period_seconds
            
            # Calculate total distance from each timeframe to the nearest hit
            for tf in test_timeframes:
                if hit_times:
                    nearest_distance = min(abs(tf - hit) for hit in hit_times)
                    total_distance += nearest_distance
                    
            # Keep track of the best offset
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_phase_offset = test_offset
        
        # Now use the best phase offset to generate the final timeframes
        period_timeframes = []
        current_time = start_time + best_phase_offset
        while current_time <= end_time:
            period_timeframes.append(float(current_time))
            current_time += period_seconds
        
        return {
            "period_seconds": float(period_seconds),
            "period_timeframes": period_timeframes
        }
    else:
        return {
            "period_seconds": float(period_seconds) if period_seconds else 0,
            "period_timeframes": []
        }

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

def extract_component_hits(audio, onset_env, component_name, start_time, end_time):
    """Extract hit times for a drum component without full pattern analysis"""
    # Convert time to frames
    start_frame = max(0, time_to_frame(start_time))
    end_frame = min(len(onset_env), time_to_frame(end_time))
    
    # Extract segment of interest
    segment_onset_env = onset_env[start_frame:end_frame]
    
    if len(segment_onset_env) == 0:
        return []
    
    # Peak Picking
    onset_frames = librosa.util.peak_pick(
        segment_onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.10,
        wait=2
    )
    
    # Convert to absolute frame indices
    onset_frames = onset_frames + start_frame
    
    # Create a list of (frame, strength) tuples
    onset_frames_with_strength = []
    for frame in onset_frames:
        if frame < len(onset_env):
            strength = onset_env[frame]
            onset_frames_with_strength.append((frame, float(strength)))
    
    # Convert frames to times and keep the strength values
    onset_times_with_strength = [(float(frame_to_time(frame)), strength) 
                            for frame, strength in onset_frames_with_strength]
    
    return onset_times_with_strength

def find_beat_defining_hits(segment, kick_beat_matches, snare_beat_matches, window=0.5):
    """
    For each beat match, check if there's a period timeframe nearby and use it as a defining hit.
    Also ensure each periodicity timeframe has a corresponding defining hit.
    
    Args:
        segment: Segment dictionary with drum_analysis already computed
        kick_beat_matches: List of (hit_time, beat_time, strength) tuples for kick
        snare_beat_matches: List of (hit_time, beat_time, strength) tuples for snare
        window: Maximum time window to consider a match
    """
    if not segment or "drum_analysis" not in segment:
        return segment
    
    # Choose whether to match to beat or hit time
    use_beat_time = True
        
    drum_analysis = segment["drum_analysis"]
    
    # Process kick pattern
    if "kick" in drum_analysis:
        kick_analysis = drum_analysis["kick"]
        period_timeframes = kick_analysis.get("period_timeframes", [])
        hit_times_with_strength = kick_analysis.get("hit_times_with_strength", [])
        
        # Initialize beat defining hits
        beat_defining_hits = []
        
        # For each kick beat match, check if there's a period timeframe nearby
        # This ensures strong hits that align with the beat are included
        for hit_time, beat_time, strength in kick_beat_matches:
            if use_beat_time:
                hit_time = beat_time
            # Check if this hit is close to any period timeframe
            is_close_to_timeframe = False
            for tf in period_timeframes:
                if abs(hit_time - tf) <= window:
                    is_close_to_timeframe = True
                    break
                    
            # If this hit is close to a period timeframe, add it
            if is_close_to_timeframe:
                beat_defining_hits.append((float(hit_time), float(strength)))
        
        # Now ensure each period timeframe has a corresponding defining hit
        # This maintains the complete grid even for timeframes without direct hits
        covered_timeframes = set()
        
        # Mark timeframes that already have a close defining hit
        for timeframe in period_timeframes:
            has_close_hit = False
            for hit_time, strength in beat_defining_hits:
                if abs(hit_time - timeframe) <= window:
                    covered_timeframes.add(timeframe)
                    has_close_hit = True
                    break
        
        # For uncovered timeframes, find the best match or use the timeframe itself
        for timeframe in period_timeframes:
            if timeframe in covered_timeframes:
                continue
                
            # Try to find a hit that's close
            best_match_time = None
            best_match_strength = 0
            min_distance = float('inf')
            
            # Check kick hits
            for hit_time, strength in hit_times_with_strength:
                distance = abs(hit_time - timeframe)
                if distance < min_distance and distance <= window:
                    min_distance = distance
                    best_match_time = hit_time
                    best_match_strength = strength
            
            # If found a match, add it
            if best_match_time is not None:
                beat_defining_hits.append((float(best_match_time), float(best_match_strength)))
            else:
                # If no match found, use the timeframe itself with medium strength
                beat_defining_hits.append((float(timeframe), 0.5))
        
        # Sort by time for consistency
        beat_defining_hits.sort(key=lambda x: x[0])
        
        # Save the beat defining hits to the kick analysis
        kick_analysis["beat_defining_hits"] = beat_defining_hits
    
    # Similarly process snare pattern using the same approach
    if "snare" in drum_analysis:
        snare_analysis = drum_analysis["snare"]
        period_timeframes = snare_analysis.get("period_timeframes", [])
        hit_times_with_strength = snare_analysis.get("hit_times_with_strength", [])
        
        # Initialize beat defining hits
        beat_defining_hits = []
        
        # For each snare beat match, check if there's a period timeframe nearby
        for hit_time, beat_time, strength in snare_beat_matches:
            if use_beat_time:
                hit_time = beat_time
            # Check if this hit is close to any period timeframe
            is_close_to_timeframe = False
            for tf in period_timeframes:
                if abs(hit_time - tf) <= window:
                    is_close_to_timeframe = True
                    break
                    
            # If this hit is close to a period timeframe, add it
            if is_close_to_timeframe:
                beat_defining_hits.append((float(hit_time), float(strength)))
        
        # Now ensure each period timeframe has a corresponding defining hit
        covered_timeframes = set()
        
        # Mark timeframes that already have a close defining hit
        for timeframe in period_timeframes:
            has_close_hit = False
            for hit_time, strength in beat_defining_hits:
                if abs(hit_time - timeframe) <= window:
                    covered_timeframes.add(timeframe)
                    has_close_hit = True
                    break
        
        # For uncovered timeframes, find the best match or use the timeframe itself
        for timeframe in period_timeframes:
            if timeframe in covered_timeframes:
                continue
                
            # Try to find a hit that's close
            best_match_time = None
            best_match_strength = 0
            min_distance = float('inf')
            
            # Check snare hits
            for hit_time, strength in hit_times_with_strength:
                distance = abs(hit_time - timeframe)
                if distance < min_distance and distance <= window:
                    min_distance = distance
                    best_match_time = hit_time
                    best_match_strength = strength
            
            # If found a match, add it
            if best_match_time is not None:
                beat_defining_hits.append((float(best_match_time), float(best_match_strength)))
            else:
                # If no match found, use the timeframe itself with medium strength
                beat_defining_hits.append((float(timeframe), 0.5))
        
        # Sort by time for consistency
        beat_defining_hits.sort(key=lambda x: x[0])
        
        # Save the beat defining hits to the snare analysis
        snare_analysis["beat_defining_hits"] = beat_defining_hits
    
    return segment

def calculate_light_strength_envelope(segment, resolution_ms=5, min_strength=0.01, snare_multi=1, max_snare_fadeout=1.5, kick_multi=0.5, max_kick_fadeout=0.8):
    """
    Calculate a combined strength envelope from kick and snare beat-defining hits.
    Uses ONLY real hits (no phantom/grid markers) for more authentic light patterns.
    - Snare hits reach 100% intensity with longer fadeouts (up to 1.5s)
    - Kick hits reach 80% intensity with medium fadeouts (up to 0.8s)
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
            
            # Calculate fade-out time (min between time to next hit or max fadeout)
            time_to_next = next_hit_time - time
            # Ensure fadeout completes before the next hit
            fadeout_time = min(time_to_next, max_snare_fadeout)
            
            # Store snare with its fadeout time
            snare_fadeouts.append((time, strength, fadeout_time))
    
    # Only create fade-outs for actual kick hits (NEW)
    for time, strength in kick_hits:
        # Check if this is a real kick hit
        if any(abs(time - rt) < 0.05 for rt in real_kick_times):
            # Find time to next defining hit (of any type)
            next_hit_time = segment_end
            for hit_time, _ in all_hits:
                if hit_time > time and hit_time < next_hit_time:
                    next_hit_time = hit_time
            
            # Calculate fade-out time (min between time to next hit or max fadeout)
            time_to_next = next_hit_time - time
            # Ensure fadeout completes before the next hit
            fadeout_time = min(time_to_next, max_kick_fadeout)
            
            # Store kick with its fadeout time
            kick_fadeouts.append((time, strength, fadeout_time))
    
    # Generate envelope points
    resolution_sec = resolution_ms / 1000
    envelope_times = []
    envelope_values = []
    
    # Start with baseline at segment start
    current_time = segment_start
    envelope_times.append(current_time)
    envelope_values.append(min_strength)
    
    # Add points for each time step
    while current_time <= segment_end:
        # Start with baseline value
        current_value = min_strength
        
        # Process kick contribution - with EXTENDED FADEOUT
        kick_contribution = min_strength
        
        # Process immediate kick influence for smoother initial peak
        for kick_time, kick_strength in kick_hits:
            if any(abs(kick_time - rt) < 0.05 for rt in real_kick_times):
                kick_time_diff = abs(kick_time - current_time)
                window_kick = 0.1  # 100ms window for immediate kick influence
                
                if kick_time_diff <= window_kick:
                    # Time falloff - closer hits have more influence
                    time_weight = 1.0 - (kick_time_diff / window_kick)
                    kick_value = kick_multi * time_weight  # Kick reaches 80% max
                    kick_contribution = max(kick_contribution, kick_value)
        
        # Process extended kick fadeouts (NEW)
        for kick_time, kick_strength, fadeout_time in kick_fadeouts:
            # Check if this time point is within the fadeout window
            if kick_time <= current_time <= (kick_time + fadeout_time):
                # Calculate how far into the fadeout we are
                fadeout_progress = (current_time - kick_time) / fadeout_time
                
                # Make sure we reach EXACTLY min_strength at the end
                if fadeout_progress >= 0.99:  # Just before the end
                    kick_value = min_strength
                else:
                    # Create a smoother decay curve
                    decay_factor = 1.0 - fadeout_progress
                    decay_factor = decay_factor ** 1.5  # Smoother curve
                    
                    # Calculate contribution with gradual falloff to min_strength
                    initial_strength = kick_multi  # 80% max intensity for kicks
                    kick_value = min_strength + (initial_strength - min_strength) * decay_factor
                
                kick_contribution = max(kick_contribution, kick_value)
        
        # Process snare influence with extended fade-out
        snare_contribution = min_strength
        
        # First check for immediate snare impact
        window_snare_immediate = 0.1  # 100ms for immediate impact
        
        for snare_time, snare_strength in snare_hits:
            # Check if this is a real snare hit
            if any(abs(snare_time - rt) < 0.05 for rt in real_snare_times):
                snare_time_diff = abs(snare_time - current_time)
                
                if snare_time_diff <= window_snare_immediate:
                    # Time falloff for immediate impact
                    time_weight = 1.0 - (snare_time_diff / window_snare_immediate)
                    immediate_value = snare_multi * time_weight  # Full strength for snare
                    snare_contribution = max(snare_contribution, immediate_value)
        
        # Then process extended snare fadeouts (ensure they reach min_strength)
        for snare_time, snare_strength, fadeout_time in snare_fadeouts:
            # Check if this time point is within the fadeout window
            if snare_time <= current_time <= (snare_time + fadeout_time):
                # Calculate how far into the fadeout we are
                fadeout_progress = (current_time - snare_time) / fadeout_time
                
                # Make sure we reach EXACTLY min_strength at the end
                if fadeout_progress >= 0.99:  # Just before the end
                    snare_value = min_strength
                else:
                    # Create a smoother decay curve
                    decay_factor = 1.0 - fadeout_progress
                    decay_factor = decay_factor ** 1.5  # Smoother curve
                    
                    # Calculate contribution with gradual falloff to min_strength
                    snare_value = min_strength + (snare_multi - min_strength) * decay_factor
                
                snare_contribution = max(snare_contribution, snare_value)
        
        # Take the maximum of kick and snare contributions
        current_value = max(kick_contribution, snare_contribution)
        
        envelope_times.append(current_time)
        envelope_values.append(current_value)
        
        # Move to next time step
        current_time += resolution_sec
    
    # Add final point if needed
    if envelope_times[-1] < segment_end:
        envelope_times.append(segment_end)
        envelope_values.append(min_strength)
    
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
            
            # Convert to milliseconds and make RELATIVE to segment start
            active_ranges.append({
                "start_ms": int((range_start - segment_start) * 1000),  # Make relative
                "end_ms": int((range_end - segment_start) * 1000),  # Make relative
                "duration_ms": int((range_end - range_start) * 1000),
                "max_value": max(envelope_values[
                    envelope_times.index(range_start):envelope_times.index(range_end)+1
                ])
            })
    
    # Convert envelope_times to be relative to segment_start
    relative_envelope_times = [t - segment_start for t in envelope_times]
    
    return {
        "times": relative_envelope_times,  # Now relative to segment start
        "values": envelope_values,
        "segment_start": segment_start,  # Keep absolute reference for context
        "segment_end": segment_end,      # Keep absolute reference for context
        "active_ranges": active_ranges   # Now with relative timestamps
    }