import os
import random
import subprocess
import threading
from .showstructurer import ShowStructurer
from services import audio_analyzer


class QueueManager:
    def __init__(self, setupfile, data_manager, qlc):
        self.queue = []
        self.ready = []
        self.analysed = []
        self.dm = data_manager
        self.qlc = qlc
        self.structurer = ShowStructurer(data_manager)

    def analyze_file(self, audio_name, filepath=None, strobes=True):
        print(f"Analyzing track {audio_name}")
        
        if not filepath:
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
        self.analyze_data(audio_name, strobes)

    def analyze_data(self, audio_name, strobes=True):
        struct_data = self.dm.get_struct_data(audio_name)
        params = audio_analyzer.segment(audio_name, struct_data)
        self.dm.update_struct_data(audio_name, params, indent=2)
        self.structurer.generate_show(audio_name, self.qlc, strobes)
    
    def sync_with_struct(self):
        self.dm.sync_with_struct()

    def analyze_queue(self, queue_folder):
        for file in os.listdir(queue_folder):
            if file.endswith(".wav"):
                audio_name = file[:-4]
                self.analyze_file(audio_name, f"{queue_folder}/{file}")
                self.queue.append(audio_name)

    def play_track(self, audio_name):
        print(f"Playing track {audio_name}")
        path = self.qlc.get_path(audio_name)
        path = self.convert_to_windows_path(path)
        return self.start_show(audio_name, path)
    
    def concurrent_analyze(self, folder, queue):
        print("--------------------------Analyzing all songs--------------------------")
        for file in queue:
            audio_name = file
            if file.endswith(".wav"):
                file_type = "wav"
            elif file.endswith(".mp3"):
                file_type = "mp3"
            print(f"AUTOANALYSIS: Analyzing track {audio_name}")
            self.analysed.append(audio_name)
            self.analyze_file(audio_name, f"{folder}/{file}{file_type}")

    def auto_play_track(self, audio_name):
        data = self.dm.get_song(audio_name)
        if data is None:
            path = None
        else:
            path = data['file']
        self.analyze_track(audio_name, path)
        path = self.qlc.get_path(audio_name)
        path = self.convert_to_windows_path(path)
        threading.Thread(target=self.start_show, args=(audio_name, path)).start()
        print("threading")

    def start_show(self, audio_name, path):
        print(f"Starting show for {audio_name}")
        command = f"/mnt/c/Windows/System32/cmd.exe /C py C:\\\\ProgramData\\\\QLCshows\\\\play_track.py play {path}"
        subprocess.run(command, shell=True)
        print(self.dm.get_song(audio_name)['file'])
        subprocess.run(f"play {self.dm.get_song(audio_name)['file']}", shell=True)
        return True

    def convert_to_windows_path(self, wsl_path):
        # Replace '/mnt/c/' with 'C:\'
        windows_path = wsl_path.replace('/mnt/c/', 'C:\\\\')
        # Replace all forward slashes with backslashes
        windows_path = windows_path.replace('/', '\\\\')
        return windows_path
    
    def choose_folder(self, folder_path):
        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        # Initialize a shuffled list of ready songs
        ready_songs = [file.split('.')[0] for file in files if self.dm.get_song(file.split('.')[0]) is not None]
        not_ready_songs = [file.split('.')[0] for file in files if self.dm.get_song(file.split('.')[0]) is None]
        
        random.shuffle(ready_songs)
        random.shuffle(not_ready_songs)

        self.queue = ready_songs + not_ready_songs

        self.concurrent_analyze(folder_path, not_ready_songs)

    def play_songs(self):
        ready = True
        print(self.queue)
        while True:
            if ready == True:
                print("Playing songs-----------------------------------------------------")
                ready = False
                song = self.queue.pop(0)
                ready = self.play_track(song)
                if len(self.queue) == 0:
                    break
