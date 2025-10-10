import os
import sys
import subprocess
import shutil
import time
import glob
import numpy as np
import librosa

def separate_drums_with_larsnet(drum_audio_path, output_dir, wiener_filter=1, device="cuda"):
    """
    Separates drum components using LarsNet with a local conda environment.
    First checks if separation has already been performed.
    
    Args:
        drum_audio_path (str): Path to the drums audio file.
        output_dir (str): Directory to save separated components.
        wiener_filter (float): α parameter for Wiener filtering (default: 1.0).
        device (str): Device to use for inference (default: "cuda").
        
    Returns:
        tuple: Dictionary of waveforms and sample rate for the drum components
    """
    print(f"--------Separating drum components using LarsNet...")
    
    # Get the base name of the file without extension for subdirectory
    base_name = os.path.splitext(os.path.basename(drum_audio_path))[0]
    
    # Create a subdirectory for this song's outputs
    song_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(song_output_dir, exist_ok=True)
    seperate_path = f"{output_dir}/drums"
    
    # Check if components already exist in the htdemucs structure
    if os.path.exists(seperate_path):
        component_paths = {}
        required_components = ['kick', 'snare', 'toms']
        all_components = required_components
        
        # Check if all required components exist
        all_required_exist = True
        
        for component in all_components:
            component_path = os.path.join(seperate_path, component, "drums.wav")
            if os.path.exists(component_path):
                component_paths[component] = component_path
                print(f"----------Found existing {component} component at {component_path}")
            elif component in required_components:
                all_required_exist = False
                print(f"----------Missing required {component} component")
        
        # If all required components exist, load them and skip separation
        if all_required_exist and len(component_paths) >= 2:
            output = {}
            sr = None
            max_length = 0
            
            # First pass: load audio and determine max length and sample rate
            for component, path in component_paths.items():
                audio, sample_rate = librosa.load(path, sr=None, mono=True)
                output[component] = audio
                
                if sr is None:
                    sr = sample_rate
                
                max_length = max(max_length, len(audio))
            
            # Second pass: resample and pad if needed
            for component in component_paths.keys():
                # Resample if needed
                if len(output[component]) != max_length:
                    output[component] = np.pad(output[component], 
                                             (0, max_length - len(output[component])), 
                                             mode='constant')
            return output, sr
    
    print(f"------------Pre-separated components not found or incomplete, performing separation...")
    
    # Determine LarsNet path
    larsnet_paths = [
        os.path.join(os.getcwd(), "src", "services", "larsnet"),
        os.path.join(os.path.dirname(os.getcwd()), "larsnet")
    ]

    print(f"------------Searching for LarsNet in {larsnet_paths}...")
    
    larsnet_dir = None
    for path in larsnet_paths:
        if os.path.exists(os.path.join(path, "separate.py")):
            larsnet_dir = path
            break
    
    if not larsnet_dir:
        raise FileNotFoundError("------------Could not find LarsNet directory with separate.py")

    print(f"------------Created output directory for this song: {song_output_dir}")
    
    # Create a temporary directory for the input drum file
    temp_input_dir = os.path.join(output_dir, "tmp_larsnet_input")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Copy the drum file to the temporary input directory
    drum_filename = os.path.basename(drum_audio_path)
    temp_drum_path = os.path.join(temp_input_dir, drum_filename)
    shutil.copy2(drum_audio_path, temp_drum_path)
    
    # Set environment variables to avoid threading issues
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    cmd = [
        sys.executable, "separate.py",
        "-i", os.path.abspath(temp_input_dir),
        "-o", os.path.abspath(song_output_dir),
        "-d", device
    ]
    
    # Add Wiener filter parameter if specified
    if wiener_filter > 0:
        cmd.extend(["-w", str(wiener_filter)])
    
    print(f"------------Running LarsNet with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=env, cwd=larsnet_dir)
        
        # Wait a moment for files to be fully written
        time.sleep(1)
        
        # Find all WAV files in the output directory
        all_wavs = glob.glob(os.path.join(song_output_dir, "**", "*.wav"), recursive=True)
        if not all_wavs:
            all_wavs = glob.glob(os.path.join(song_output_dir, "*.wav"))
        
        # Search for kick, snare, and toms files in several possible locations
        possible_paths = {
            'kick': [
                os.path.join(song_output_dir, "kick", "drums.wav"),
            ],
            'snare': [
                os.path.join(song_output_dir, "snare", "drums.wav")
            ],
            'toms': [
                os.path.join(song_output_dir, "toms", "drums.wav")
            ],
        }
        
        # Find the actual paths from the possibilities or by searching
        output_paths = {}
        for component, paths in possible_paths.items():
            # Try the predefined paths first
            found = False
            for path in paths:
                if os.path.exists(path):
                    output_paths[component] = path
                    print(f"Found {component} at {path}")
                    found = True
                    break
            
            # If not found in predefined paths, search through all WAVs
            if not found:
                for wav_path in all_wavs:
                    if component.lower() in os.path.basename(wav_path).lower():
                        output_paths[component] = wav_path
                        print(f"Found {component} at {wav_path}")
                        found = True
                        break
            
            # If still not found, it will be missing from output_paths
            
        # Load kick and snare using librosa to handle various audio formats
        kick_audio, kick_sr = librosa.load(output_paths['kick'], sr=None, mono=True)
        snare_audio, snare_sr = librosa.load(output_paths['snare'], sr=None, mono=True)
        
        # Make sure they're the same sample rate
        if kick_sr != snare_sr:
            # Resample the one with higher sample rate to match the lower one
            sr = min(kick_sr, snare_sr)
            if kick_sr > sr:
                kick_audio = librosa.resample(kick_audio, orig_sr=kick_sr, target_sr=sr)
            if snare_sr > sr:
                snare_audio = librosa.resample(snare_audio, orig_sr=snare_sr, target_sr=sr)
        else:
            sr = kick_sr
        
        # Make sure they're the same length
        max_length = max(len(kick_audio), len(snare_audio))
        if len(kick_audio) < max_length:
            kick_audio = np.pad(kick_audio, (0, max_length - len(kick_audio)), mode='constant')
        if len(snare_audio) < max_length:
            snare_audio = np.pad(snare_audio, (0, max_length - len(snare_audio)), mode='constant')
        
        # Add toms if available
        if 'toms' in output_paths:
            try:
                toms_audio, toms_sr = librosa.load(output_paths['toms'], sr=sr, mono=True)
                # Make sure toms are the same length
                if len(toms_audio) < max_length:
                    toms_audio = np.pad(toms_audio, (0, max_length - len(toms_audio)), mode='constant')
                elif len(toms_audio) > max_length:
                    toms_audio = toms_audio[:max_length]
            except Exception as e:
                pass
        
        # Prepare output dictionary
        output = {}
        output['kick'] = kick_audio
        output['snare'] = snare_audio
        if 'toms' in output_paths:
            output['toms'] = toms_audio

        # Clean up temporary input directory
        if os.path.exists(temp_input_dir):
            try:
                shutil.rmtree(temp_input_dir)
            except:
                print(f"------------Warning: Failed to clean up temporary directory {temp_input_dir}")

        print(f"--------LarsNet separation completed successfully.")
        return output, sr
        
    except Exception as e:
        print(f"------------Error running LarsNet: {e}")
        # Clean up temporary input directory
        if os.path.exists(temp_input_dir):
            try:
                shutil.rmtree(temp_input_dir)
            except:
                pass
            
        # Fall back to original drums audio
        try:
            print("--------------Using original drums as fallback due to error")
            audio, sr = librosa.load(drum_audio_path, sr=None, mono=True)
            return {'full_drums': audio}, sr
        except:
            print("--------------Failed to load original drums. Returning empty audio.")
            return {'full_drums': np.zeros(1000)}, 22050