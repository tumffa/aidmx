import os
import subprocess
import shutil
import time
import glob
import numpy as np
import scipy.io.wavfile as wav
import librosa

def separate_drums_with_larsnet(drum_audio_path, output_dir, wiener_filter=1, device="cuda"):
    """
    Separates drum components using LarsNet with a local conda environment.
    
    Args:
        drum_audio_path (str): Path to the drums audio file.
        output_dir (str): Directory to save separated components.
        wiener_filter (float): α parameter for Wiener filtering (default: 1.0).
        device (str): Device to use for inference (default: "cuda").
        
    Returns:
        tuple: (waveform, sample_rate) for the combined drums
    """
    print(f"Separating drum components using LarsNet on {device}...")
    
    # Determine LarsNet path
    larsnet_paths = [
        os.path.join(os.getcwd(), "src", "services", "larsnet"),
        os.path.join(os.path.dirname(os.getcwd()), "larsnet")
    ]

    print(f"Searching for LarsNet in {larsnet_paths}...")
    
    larsnet_dir = None
    for path in larsnet_paths:
        if os.path.exists(os.path.join(path, "separate.py")):
            larsnet_dir = path
            break
    
    if not larsnet_dir:
        raise FileNotFoundError("Could not find LarsNet directory with separate.py")
    
    # Get the base name of the file without extension for subdirectory
    base_name = os.path.splitext(os.path.basename(drum_audio_path))[0]
    
    # Create a subdirectory for this song's outputs
    song_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(song_output_dir, exist_ok=True)
    print(f"Created output directory for this song: {song_output_dir}")
    
    # Check for conda environment in the LarsNet directory
    env_dir = os.path.join(larsnet_dir, "env")
    if not os.path.exists(env_dir):
        print(f"Warning: Could not find conda environment at {env_dir}")
        print("Using system Python instead. This might cause compatibility issues.")
        use_conda = False
    else:
        use_conda = True
    
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
    
    # Build command for LarsNet separation
    separate_script = os.path.join(larsnet_dir, "separate.py")
    
    if use_conda:
        # Using conda environment
        cmd = [
            "conda", "run", "--prefix", env_dir,
            "python", separate_script,
            "-i", temp_input_dir,
            "-o", song_output_dir,  # Use the song-specific output directory
            "-d", device
        ]
    else:
        # Using system Python - IMPORTANT: Change directory to LarsNet dir
        cmd = [
            "cd", larsnet_dir, "&&",
            "python", "separate.py",
            "-i", os.path.abspath(temp_input_dir),
            "-o", os.path.abspath(song_output_dir),  # Use the song-specific output directory
            "-d", device
        ]
    
    # Add Wiener filter parameter if specified
    if wiener_filter > 0:
        cmd.extend(["-w", str(wiener_filter)])
    
    print(f"Running LarsNet with command: {' '.join(cmd)}")
    
    try:
        if not use_conda:
            # For system Python, we need to run the command in a shell and change directory
            cmd_str = " ".join(cmd)
            subprocess.run(cmd_str, shell=True, check=True, env=env)
        else:
            subprocess.run(cmd, check=True, env=env)
        
        # Wait a moment for files to be fully written
        time.sleep(1)
        
        # Find all WAV files in the output directory
        print(f"Searching for component files in {song_output_dir}...")
        all_wavs = glob.glob(os.path.join(song_output_dir, "**", "*.wav"), recursive=True)
        if not all_wavs:
            all_wavs = glob.glob(os.path.join(song_output_dir, "*.wav"))
        
        print(f"Found {len(all_wavs)} WAV files in output directory")
        
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
        
        # If we didn't find kick and snare, we can't proceed
        if 'kick' not in output_paths or 'snare' not in output_paths:
            print("Could not find kick and/or snare components")
            # Fall back to original drums audio
            audio, sr = librosa.load(drum_audio_path, sr=None, mono=True)
            print(f"Using original drums as fallback: shape={audio.shape}, sr={sr}")
            return audio, sr
            
        # Load kick and snare using librosa to handle various audio formats
        kick_audio, kick_sr = librosa.load(output_paths['kick'], sr=None, mono=True)
        snare_audio, snare_sr = librosa.load(output_paths['snare'], sr=None, mono=True)
        
        # Make sure they're the same sample rate
        if kick_sr != snare_sr:
            print(f"Warning: Kick and snare have different sample rates ({kick_sr} vs {snare_sr})")
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
                print(f"Error loading toms audio: {e}. Continuing without toms.")

        # Resample
        

        output = {}
        output['kick'] = kick_audio
        output['snare'] = snare_audio
        if 'toms' in output_paths:
            output['toms'] = toms_audio

        return output, sr
        
    except Exception as e:
        print(f"Error running LarsNet: {e}")
        # Clean up temporary input directory
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
            
        # Fall back to original drums audio
        try:
            print("Using original drums as fallback due to error")
            audio, sr = librosa.load(drum_audio_path, sr=None, mono=True)
            return audio, sr
        except:
            print("Failed to load original drums. Returning empty audio.")
            return np.zeros(1000), 22050