import os
import threading
import time
import pygame
from src.services.showstructurer import ShowStructurer
from src.services.audio_analysis import audio_analysis
from src.services.ola_dmx_controller import play_dmx_sequence

class QueueManager:
    def __init__(self, setupfile, data_manager, qlc):
        self.queue = []
        self.ready = []
        self.dm = data_manager
        self.qlc = qlc
        self.structurer = ShowStructurer(data_manager)

    def analyze_file(self, audio_name, strobes=False, simple=False, delay=0):
        """Starts the file analysis process and show generation process.

        Args:
            audio_name (str): The name of the audio file to analyze.
            strobes (bool, optional): Whether to include strobes in the show generation. Defaults to False.
            simple (bool, optional): Whether to use simple mode (only a simple color chaser). Defaults to False.
            delay (int, optional): Delay in ms before show start to allow for i.e. song track command to begin. Defaults to 600.
        """
        print(f"\nProcessing {audio_name} with strobes={strobes}, simple={simple}, delay={delay} seconds")

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
    
        self.dm.extract_data(audio_name, os.path.abspath(filepath))
        self.analyze_data(audio_name, delay, strobes, simple)
        print(f"Finished\n")

    def analyze_data(self, audio_name, delay, strobes=False, simple=False):
        print(f"--Analyzing data")
        struct_data = self.dm.get_struct_data(audio_name)
        params = audio_analysis.segment(audio_name, struct_data)
        self.dm.update_struct_data(audio_name, params, indent=2)
        
        print(f"--Generating show")
        self.qlc.create_copy(audio_name)
        scripts_tuple = self.structurer.generate_show(audio_name, self.qlc, strobes=strobes, simple=simple)
        if self.structurer.dmx_controller == "qlc":
            scripts, function_names = scripts_tuple
            self.qlc.add_track(scripts, audio_name, function_names)
        elif self.structurer.dmx_controller == "ola":
            frame_delays_ms, dmx_frames = scripts_tuple
            song_path = struct_data.get("filepath")
            if not song_path or not os.path.exists(song_path):
                print(f"Audio path not found for {audio_name}: {song_path}")
                return
            self.song_playback_and_ola(song_path, frame_delays_ms, dmx_frames, delay=delay, universe=1)

    def song_playback_and_ola(self, song_path, frame_delays_ms, dmx_frames, delay, universe):
        pygame.mixer.init()
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()

        # Wait until audio is actually playing
        while not pygame.mixer.music.get_busy():
            time.sleep(0.01)

        time.sleep(delay)

        dmx_thread = threading.Thread(target=play_dmx_sequence, args=(frame_delays_ms, dmx_frames, universe))
        dmx_thread.start()
        dmx_thread.join()
        pygame.mixer.music.stop()
        
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
