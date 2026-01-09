import os
import threading
import time
from src.services.showstructurer import ShowStructurer
from src.services.ola_dmx_controller import play_dmx_sequence, stop_current_playback

class QueueManager:
    def __init__(self, setupfile, data_manager, qlc=None):
        self.queue = []
        self.ready = []
        self.dm = data_manager
        self.qlc = qlc
        self.structurer = ShowStructurer(data_manager)
        # Cancellation state shared across tasks
        self.cancel_event = threading.Event()
        self._playback_thread = None

    def analyze_file(self, audio_name, strobes, simple, qlc_delay, qlc_lag):
        """Starts the file analysis process and show generation process.

        Args:
            audio_name (str): The name of the audio file to analyze.
            strobes (bool, optional): Whether to include strobes in the show generation. Defaults to False.
            simple (bool, optional): Whether to use simple mode (only a simple color chaser). Defaults to False.
            ola_delay (int, optional): Delay in ms before show start to allow for i.e. song track command to begin. Defaults to 600.
            qlc_delay (int, optional): Delay in ms before show start to allow for i.e. song track command to begin. Defaults to 0.
        """
        print(
            f"\nProcessing {audio_name} with strobes={strobes}, simple={simple}, qlc_delay={qlc_delay} sec, qlc_lag={qlc_lag}")

        # Clear any previous cancellations
        self.cancel_event.clear()

        # Check for both .mp3 and .wav files
        mp3_path = "{}/songs/{}.mp3".format(self.dm.return_path("data"), audio_name)
        wav_path = "{}/songs/{}.wav".format(self.dm.return_path("data"), audio_name)
        
        if os.path.exists(mp3_path):
            filepath = mp3_path
            print(f"Using default path: {filepath}")
        elif os.path.exists(wav_path):
            filepath = wav_path
            print(f"Using default path: {filepath}")
        else:
            print(f"No file found for {audio_name} in default paths.")
            return
    
        if self.cancel_event.is_set():
            print("--Analysis canceled before extraction")
            return

        self.dm.extract_data(audio_name, os.path.abspath(filepath))

        if self.cancel_event.is_set():
            print("--Analysis canceled before generation")
            return

        self.generate(audio_name, strobes, simple, qlc_delay, qlc_lag)
        print(f"Finished\n")

    def generate(self, audio_name, strobes, simple, qlc_delay, qlc_lag):
        print(f"--Generating show with strobes={strobes}, simple={simple}, qlc_delay={qlc_delay} sec, qlc_lag={qlc_lag}")
        self.qlc.create_copy(audio_name)
        scripts_dict = self.structurer.generate_show(
            audio_name, 
            self.qlc, 
            strobes=strobes, 
            simple=simple, 
            qlc_delay=qlc_delay,
            qlc_lag=qlc_lag)

        if self.cancel_event.is_set():
            print("--Analysis canceled after generation")
            return

        scripts = scripts_dict["qlc"]["scripts"]
        function_names = scripts_dict["qlc"]["function_names"]
        self.qlc.add_track(audio_name, scripts, function_names)

        frame_delays_ms, dmx_frames = scripts_dict["ola"]["frame_delays_ms"], scripts_dict["ola"]["dmx_frames"]
        self.dm.save_ola_sequence(audio_name, frame_delays_ms, dmx_frames)

    def play_ola_show(self, audio_name, delay, universe, start_at_sec=0.0):
        print(f"--Playing OLA show for {audio_name} with delay {delay} sec on universe {universe}, start_at={start_at_sec:.2f}s")

        # Clear any previous cancellations
        self.cancel_event.clear()

        struct_data = self.dm.get_struct_data(audio_name)
        song_path = struct_data.get("filepath")
        if not song_path or not os.path.exists(song_path):
            print(f"Audio path not found for {audio_name}: {song_path}")
            return
        
        frame_delays_ms, dmx_frames = self.dm.load_ola_sequence(audio_name)
        print(f"--Loaded OLA sequence from {song_path}")

        self.song_playback_and_ola(song_path, frame_delays_ms, dmx_frames, delay=delay, universe=universe, start_at_sec=start_at_sec)

    def song_playback_and_ola(self, song_path, frame_delays_ms, dmx_frames, delay, universe, start_at_sec=0.0):
        import pygame
        print(f"--Starting playback of {song_path} (seek to {start_at_sec:.2f}s)")
        pygame.mixer.init()
        pygame.mixer.music.load(song_path)
        # Try to start at the requested timestamp
        try:
            pygame.mixer.music.play(start=max(0.0, float(start_at_sec)))
        except Exception:
            pygame.mixer.music.play()
            try:
                pygame.mixer.music.set_pos(max(0.0, float(start_at_sec)))
            except Exception:
                # Fallback: start from 0 if seeking not supported
                pass

        # Wait until audio is actually playing
        while not pygame.mixer.music.get_busy():
            time.sleep(0.01)

        # Allow cancel during start delay
        waited = 0.0
        while waited < float(delay):
            if self.cancel_event.is_set():
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                print("--Playback canceled before DMX start")
                return
            time.sleep(0.05)
            waited += 0.05

        # Slice DMX sequence to start at the same timestamp
        start_ms = max(0, int(start_at_sec * 1000))
        if frame_delays_ms and dmx_frames:
            cum = 0
            idx = None
            for i, d in enumerate(frame_delays_ms):
                cum += int(d)
                if cum >= start_ms:
                    idx = i
                    break
            if idx is None:
                print("--Start timestamp beyond DMX sequence length; nothing to play")
                pygame.mixer.music.stop()
                return

            ms_before = cum - int(frame_delays_ms[idx])
            overshoot = start_ms - ms_before
            first_delay = max(0, int(frame_delays_ms[idx]) - overshoot)

            sliced_delays = [first_delay] + list(frame_delays_ms[idx+1:])
            sliced_frames = list(dmx_frames[idx:])
        else:
            sliced_delays = frame_delays_ms
            sliced_frames = dmx_frames

        print(f"--Starting DMX sequence playback with OLA from {start_at_sec:.2f}s")
        self._playback_thread = threading.Thread(target=play_dmx_sequence, args=(sliced_delays, sliced_frames, universe), daemon=True)
        self._playback_thread.start()

        # Monitor for cancellation while DMX is playing
        try:
            while self._playback_thread.is_alive():
                if self.cancel_event.is_set():
                    print("--Playback cancel requested; stopping DMX and audio")
                    stop_current_playback()
                    try:
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                    return
                time.sleep(0.05)
        finally:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def request_cancel_analysis(self):
        """Signal cancellation for the current analysis task."""
        print("--Cancel requested for analysis")
        self.cancel_event.set()

    def request_cancel_playback(self):
        """Signal cancellation for current playback and stop DMX/audio."""
        print("--Cancel requested for playback")
        self.cancel_event.set()
        # Attempt to stop DMX immediately if running
        try:
            stop_current_playback()
        except Exception:
            pass
        # Stop audio if mixer is active
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass
        
    def sync_with_struct(self):
        self.dm.sync_with_struct()

    def analyze_queue(self, queue_folder):
        for file in os.listdir(queue_folder):
            if file.endswith(".wav"):
                audio_name = file[:-4]
                self.analyze_file(audio_name, f"{queue_folder}/{file}")
                self.queue.append(audio_name)
    
    def concurrent_analyze(self, folder):
        """ Analyze all audio files in a specified folder

        Args:
            folder (str): The name of the folder containing audio files to analyze.
        """
        print(f"--------------------------Analyzing {folder}--------------------------")
        path = self.dm.return_path("data") / folder
        songs = [file for file in os.listdir(path) if file.endswith(".mp3") or file.endswith(".wav")]
        song_amount = len(songs)
        i = 1
        print(f"Analyzing {song_amount} songs in {path}")
        for file in os.listdir(path):
            print(f"Analyzing {i}/{song_amount} - {file}")
            audio_name = file.split(".")[0]
            audio_path = os.path.join(path, file)
            print(f"Analyzing {audio_name} in {audio_path}")
            self.analyze_file(audio_name, audio_path, strobes=False)
            i += 1
        print(f"-----------------------Finished analyzing {folder}------------------------")

    def merge_shows(self, name, shows):
        """
        Merge multiple shows into a single show.
        
        Args:
            name (str): The name for the merged show
            shows (list or str): Either a list of show names OR the name of a folder containing audio files
        """
        if isinstance(shows, str):
            # Treat as folder name relative to data path
            folder_path = self.dm.return_path("data") / shows
            print(f"Looking for shows in folder: {folder_path}")
            
            show_names = []
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".wav") or file.endswith(".mp3"):
                        show_name = file.split(".")[0]
                        try:
                            # Check if we can get duration for this show
                            self.dm.get_song_length(show_name)
                            show_names.append(show_name)
                        except Exception as e:
                            print(f"Warning: Cannot get duration for {show_name}: {e}, skipping")
                
                if not show_names:
                    print(f"No valid shows found in folder {shows}")
                    return
                    
                print(f"Found {len(show_names)} valid shows in folder {shows}")
                shows = show_names
            else:
                print(f"Folder {folder_path} not found")
                return
        
        print(f"Merging shows into {name}:", shows)
        show_data = []
        for show in shows:
            try:
                duration = self.dm.get_song_length(show)
                show_data.append({'name': show, "duration": duration})
            except Exception as e:
                print(f"Error getting duration for {show}: {e}, skipping")
        
        if not show_data:
            print("No valid shows to merge")
            return
            
        self.qlc.merge_shows(name, show_data)
