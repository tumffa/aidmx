import os
import librosa
import numpy as np

# Function to convert frame index to time in seconds
def frame_to_time(frame_idx):
    return librosa.frames_to_time(frame_idx, sr=22050, hop_length=512)

# Function to convert time to frame index
def time_to_frame(t):
    return librosa.time_to_frames(t, sr=22050, hop_length=512)
    
def analyze_drum_beat_pattern(demix_path, beats=None, segments=None):
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
    
    print(f"--------Calculating dimmer scaling function - loading demixed drum tracks from {demix_path}")
    try:
        kick_audio, sr_kick = librosa.load(kick_path, sr=sr, mono=True)
        print(f"----------Found kick at {kick_path}")
    except Exception as e:
        print(f"----------Error loading kick track: {e}")
        kick_audio = np.zeros(1000)  # Fallback empty audio
    
    try:
        snare_audio, sr_snare = librosa.load(snare_path, sr=sr, mono=True)
        print(f"----------Found snare at {snare_path}")
    except Exception as e:
        print(f"----------Error loading snare track: {e}")
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
        
        # Find beat snare/kick matches
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

def match_beats_with_drum_hits(beats, kick_hits, snare_hits, window=0.15):
    """
    Match beats with drum hits based on proximity and hit strength.
    
    Args:
        beats: List of beat timeframes (in seconds)
        kick_hits: List of (time, strength) tuples for kick hits
        snare_hits: List of (time, strength) tuples for snare hits
        window: Time window (in seconds) to consider a beat matching with a hit
    
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
