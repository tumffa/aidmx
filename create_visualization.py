import json
import numpy as np
import cv2
import os
from pathlib import Path

def visualize_segments(json_path, audio_path, output_path="output_visualization.mp4"):
    """
    Create a visualization of drum patterns from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing segment data
        audio_path: Path to the audio file (for duration reference)
        output_path: Path where the output video will be saved
    """
    # Load segment data from JSON
    print(f"Loading segment data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    if not segments:
        print("No segments found in the JSON file.")
        return
    
    # Get strobe sections from onset_parts if available
    onset_parts = data.get("onset_parts", [])
    strobe_sections = []
    if onset_parts:
        print(f"Found {len(onset_parts)} strobe sections")
        for start, end in onset_parts:
            strobe_sections.append((float(start), float(end)))
    
    # Get beat times if available
    beats = data.get("beats", [])
    if beats:
        print(f"Found {len(beats)} beats")
        # Convert beats to float in case they're stored as strings
        beats = [float(beat) for beat in beats]
    
    # Video parameters
    width, height = 1920, 1080  # 1080p resolution
    fps = 30
    
    # Calculate total duration from the last segment's end time
    total_duration = max(segment["end"] for segment in segments)
    total_frames = int(total_duration * fps) + 60  # Add 2 seconds buffer
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Bar style parameters
    max_height = 400  # Maximum height of bars
    bar_width = 10  # Width of each hit bar
    kick_color = (0, 0, 255)  # Red for kicks (BGR)
    snare_color = (0, 255, 0)  # Green for snares (BGR)
    strobe_color = (255, 255, 255)  # White for strobe sections (BGR)
    beat_color = (255, 165, 0)  # Orange for beat markers (BGR)
    beat_match_color = (255, 255, 255)  # White outline for beat-matched hits
    beat_match_outline_thickness = 2  # Thickness of the outline
    kick_period_color = (128, 0, 255)  # Purple for kick period markers (BGR)
    snare_period_color = (0, 255, 255)  # Yellow for snare period markers (BGR)
    
    # Create frames for each time point
    print(f"Generating {total_frames} frames...")
    for frame_idx in range(total_frames):
        current_time = frame_idx / fps
        
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Find the current segment
        current_segment = None
        for segment in segments:
            if segment["start"] <= current_time < segment["end"]:
                current_segment = segment
                break
        
        if current_segment:
            # Segment info
            label = current_segment.get("label", "Unknown")
            segment_start = current_segment["start"]
            segment_end = current_segment["end"]
            segment_progress = (current_time - segment_start) / (segment_end - segment_start)
            
            # Display segment info at the top
            cv2.putText(frame, f"Segment: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display time information
            time_text = f"Time: {current_time:.2f}s / {segment_end:.2f}s"
            cv2.putText(frame, time_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            # Display progress bar
            progress_width = int(segment_progress * (width - 100))
            cv2.rectangle(frame, (50, 110), (50 + progress_width, 120), (100, 100, 255), -1)
            cv2.rectangle(frame, (50, 110), (width - 50, 120), (100, 100, 100), 1)
            
            # Get drum analysis data
            if "drum_analysis" in current_segment:
                drum_data = current_segment["drum_analysis"]
                
                # BPM information
                kick_tempo = drum_data.get("kick", {}).get("tempo_bpm", 0)
                snare_tempo = drum_data.get("snare", {}).get("tempo_bpm", 0)

                # Extract beat match timeframes
                kick_beat_matches = []
                if "kick" in drum_data and "beat_matches" in drum_data["kick"]:
                    # Extract just the timeframes from beat_matches
                    kick_beat_matches = [match[0] for match in drum_data["kick"]["beat_matches"]]
                
                snare_beat_matches = []
                if "snare" in drum_data and "beat_matches" in drum_data["snare"]:
                    # Extract just the timeframes from beat_matches
                    snare_beat_matches = [match[0] for match in drum_data["snare"]["beat_matches"]]
                
                # Display BPM info
                if kick_tempo > 0:
                    cv2.putText(frame, f"Kick BPM: {kick_tempo:.1f}", (width - 300, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, kick_color, 2)
                if snare_tempo > 0:
                    cv2.putText(frame, f"Snare BPM: {snare_tempo:.1f}", (width - 300, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, snare_color, 2)
                    
                # Extract period timeframes for kick and snare
                kick_period_timeframes = []
                if "kick" in drum_data and "period_timeframes" in drum_data["kick"]:
                    kick_period_timeframes = drum_data["kick"]["period_timeframes"]
                
                snare_period_timeframes = []
                if "snare" in drum_data and "period_timeframes" in drum_data["snare"]:
                    snare_period_timeframes = drum_data["snare"]["period_timeframes"]
                
                # Render kick hits
                if "kick" in drum_data and "hit_times_with_strength" in drum_data["kick"]:
                    kick_hits = drum_data["kick"]["hit_times_with_strength"]
                    for time, strength in kick_hits:
                        # Only render hits within the current segment and +/-2 seconds from current time
                        if segment_start <= time < segment_end and (current_time - 2) <= time <= (current_time + 2):
                            # Position the bar based on time
                            x_pos = int(width/2 + (time - current_time) * width/4)
                            if 0 <= x_pos < width:
                                # Height based on strength (normalize between 0.2 and 1.0 for better visibility)
                                normalized_strength = 0.2 + 0.8 * min(1.0, strength)
                                bar_height = int(normalized_strength * max_height)
                                
                                # Draw the kick bar
                                cv2.rectangle(frame, 
                                             (x_pos - bar_width//2, height//2 + 10), 
                                             (x_pos + bar_width//2, height//2 + 10 + bar_height), 
                                             kick_color, -1)
                                
                                # Add white outline if this kick has a beat match
                                if time in kick_beat_matches:
                                    cv2.rectangle(frame, 
                                                 (x_pos - bar_width//2, height//2 + 10), 
                                                 (x_pos + bar_width//2, height//2 + 10 + bar_height), 
                                                 beat_match_color, beat_match_outline_thickness)
                
                # Render snare hits
                if "snare" in drum_data and "hit_times_with_strength" in drum_data["snare"]:
                    snare_hits = drum_data["snare"]["hit_times_with_strength"]
                    for time, strength in snare_hits:
                        # Only render hits within the current segment and +/-2 seconds from current time
                        if segment_start <= time < segment_end and (current_time - 2) <= time <= (current_time + 2):
                            # Position the bar based on time
                            x_pos = int(width/2 + (time - current_time) * width/4)
                            if 0 <= x_pos < width:
                                # Height based on strength (normalize between 0.2 and 1.0 for better visibility)
                                normalized_strength = 0.2 + 0.8 * min(1.0, strength)
                                bar_height = int(normalized_strength * max_height)
                                
                                # Draw the snare bar (going upward)
                                cv2.rectangle(frame, 
                                             (x_pos - bar_width//2, height//2 - 10), 
                                             (x_pos + bar_width//2, height//2 - 10 - bar_height), 
                                             snare_color, -1)
                                
                                # Add white outline if this snare has a beat match
                                if time in snare_beat_matches:
                                    cv2.rectangle(frame, 
                                                 (x_pos - bar_width//2, height//2 - 10), 
                                                 (x_pos + bar_width//2, height//2 - 10 - bar_height), 
                                                 beat_match_color, beat_match_outline_thickness)
                                    
                # Render kick period timeframe indicators
                for time in kick_period_timeframes:
                    # Only render timeframes within the current segment and +/-2 seconds from current time
                    if segment_start <= time < segment_end and (current_time - 2) <= time <= (current_time + 2):
                        # Position the indicator based on time
                        x_pos = int(width/2 + (time - current_time) * width/4)
                        if 0 <= x_pos < width:
                            # Draw a downward triangle marker for kick periods
                            triangle_size = 8
                            pts = np.array([
                                [x_pos, height//2 + 25],  # Bottom point
                                [x_pos - triangle_size, height//2 + 25 - triangle_size],  # Top left
                                [x_pos + triangle_size, height//2 + 25 - triangle_size]   # Top right
                            ], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.fillPoly(frame, [pts], kick_period_color)
                
                # Render snare period timeframe indicators
                for time in snare_period_timeframes:
                    # Only render timeframes within the current segment and +/-2 seconds from current time
                    if segment_start <= time < segment_end and (current_time - 2) <= time <= (current_time + 2):
                        # Position the indicator based on time
                        x_pos = int(width/2 + (time - current_time) * width/4)
                        if 0 <= x_pos < width:
                            # Draw an upward triangle marker for snare periods
                            triangle_size = 8
                            pts = np.array([
                                [x_pos, height//2 - 25],  # Top point
                                [x_pos - triangle_size, height//2 - 25 + triangle_size],  # Bottom left
                                [x_pos + triangle_size, height//2 - 25 + triangle_size]   # Bottom right
                            ], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.fillPoly(frame, [pts], snare_period_color)
            
            # Draw center line
            cv2.line(frame, (0, height//2), (width, height//2), (50, 50, 50), 2)
            
            # Vertical timeline markers
            for i in range(-2, 3):
                marker_x = int(width/2 + i * width/4)
                if 0 <= marker_x < width:
                    # Draw time marker line
                    cv2.line(frame, (marker_x, height//2 - 20), (marker_x, height//2 + 20), (70, 70, 70), 1)
                    # Add time label
                    time_label = f"{i:+d}s" if i != 0 else "Now"
                    cv2.putText(frame, time_label, (marker_x - 15, height//2 + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                               
            # Draw legend for visualization elements
            legend_x = width - 400
            cv2.putText(frame, "Legend:", (legend_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # ... existing legend entries ...
            
            # Beat marker legend
            cv2.rectangle(frame, (legend_x, 160), (legend_x + 20, 180), beat_color, -1)
            cv2.putText(frame, "Beat", (legend_x + 30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Beat match legend 
            beat_match_legend_y = 190
            cv2.rectangle(frame, (legend_x, beat_match_legend_y), (legend_x + 20, beat_match_legend_y + 20), snare_color, -1)
            cv2.rectangle(frame, (legend_x, beat_match_legend_y), (legend_x + 20, beat_match_legend_y + 20), 
                         beat_match_color, beat_match_outline_thickness)
            cv2.putText(frame, "Beat Match", (legend_x + 30, beat_match_legend_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Kick period marker legend
            kick_period_y = 220
            # Draw triangle for kick period legend
            tri_pts = np.array([
                [legend_x + 10, kick_period_y + 10],
                [legend_x, kick_period_y],
                [legend_x + 20, kick_period_y]
            ], np.int32)
            tri_pts = tri_pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [tri_pts], kick_period_color)
            cv2.putText(frame, "Kick Period", (legend_x + 30, kick_period_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Snare period marker legend
            snare_period_y = 250
            # Draw triangle for snare period legend
            tri_pts = np.array([
                [legend_x + 10, snare_period_y],
                [legend_x, snare_period_y + 10],
                [legend_x + 20, snare_period_y + 10]
            ], np.int32)
            tri_pts = tri_pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [tri_pts], snare_period_color)
            cv2.putText(frame, "Snare Period", (legend_x + 30, snare_period_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)   
        
        # Render beat markers
        if beats:
            for beat_time in beats:
                # Only render beats within the current 4-second window
                if (current_time - 2) <= beat_time <= (current_time + 2):
                    # Position the beat marker based on time
                    x_pos = int(width/2 + (beat_time - current_time) * width/4)
                    if 0 <= x_pos < width:
                        # Draw a small square at the beat position
                        square_size = 8
                        cv2.rectangle(frame, 
                                     (x_pos - square_size//2, height//2 - square_size//2), 
                                     (x_pos + square_size//2, height//2 + square_size//2), 
                                     beat_color, -1)
                        
        # Render strobe sections
        for start_time, end_time in strobe_sections:
            # Check if the strobe section is visible in the current 4-second window
            if (current_time - 2 <= start_time <= current_time + 2 or 
                current_time - 2 <= end_time <= current_time + 2 or
                (start_time <= current_time - 2 and end_time >= current_time + 2)):
                
                # Calculate visible portion of strobe section
                visible_start = max(start_time, current_time - 2)
                visible_end = min(end_time, current_time + 2)
                
                # Calculate x positions
                start_x = int(width/2 + (visible_start - current_time) * width/4)
                end_x = int(width/2 + (visible_end - current_time) * width/4)
                
                if start_x < width and end_x >= 0:
                    # Adjust to visible area
                    start_x = max(0, start_x)
                    end_x = min(width, end_x)
                    
                    # Draw strobe indicator box
                    strobe_height = 30
                    cv2.rectangle(frame, 
                                 (start_x, height//2 - strobe_height//2), 
                                 (end_x, height//2 + strobe_height//2), 
                                 strobe_color, 2)
                    
                    # Fill with semi-transparent white
                    overlay = frame.copy()
                    cv2.rectangle(overlay, 
                                 (start_x, height//2 - strobe_height//2), 
                                 (end_x, height//2 + strobe_height//2), 
                                 strobe_color, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Add strobe label if section is wide enough
                    if end_x - start_x > 100:
                        cv2.putText(frame, "STROBE", 
                                   (start_x + 10, height//2 + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, strobe_color, 1)
                        
        # Write the frame
        out.write(frame)
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_path}")
    
    # Add audio to video if ffmpeg is available
    try:
        output_with_audio = os.path.splitext(output_path)[0] + "_with_audio.mp4"
        import subprocess
        command = [
            'ffmpeg',
            '-y',
            '-i', output_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            output_with_audio
        ]
        print("Adding audio to video...")
        subprocess.run(command, check=True)
        print(f"Video with audio saved to {output_with_audio}")
    except Exception as e:
        print(f"Could not add audio to video: {e}")
        print("Video without audio is still available.")
    
if __name__ == "__main__":
    path = Path(os.getcwd())
    song = "lostinstatic" # enter the song name here
    JSON_PATH = path / f"aidmx/struct/{song}.json"
    AUDIO_PATH = path / f"aidmx/data/songs/{song}.wav"
    OUTPUT_PATH = path / f"aidmx/{song}_visualization.mp4"
    
    visualize_segments(JSON_PATH, AUDIO_PATH, OUTPUT_PATH)