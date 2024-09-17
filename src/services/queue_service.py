from .qlc_service import QLCHandler
from .showstructurer import ShowStructurer
from services import audio_analyzer
import os
import random
import subprocess
import threading

class QueueManager:
    def __init__(self, setupfile, data_manager, qlc):
        self.queue = []
        self.ready = []
        self.analysed = []
        self.dm = data_manager
        self.qlc = qlc
        self.structurer = ShowStructurer(data_manager)

    def analyze_file(self, audio_name, filepath):
        print(f"Analyzing track {audio_name}")
        if not filepath:
            filepath = "{}/songs/{}.mp3".format(self.dm.return_path("data"), audio_name)
            print(f"Using default path: {filepath}")
        self.dm.extract_data(audio_name, os.path.abspath(filepath))
        self.analyze_data(audio_name)

    def analyze_data(self, audio_name):
        struct_data = self.dm.get_struct_data(audio_name)
        params = audio_analyzer.segment(audio_name, struct_data)
        self.dm.update_struct_data(audio_name, params, indent=2)
        self.structurer.generate_show(audio_name, self.qlc)
    
    def sync_with_struct(self):
        self.dm.sync_with_struct()

    def analyze_queue(self, queue_folder):
        for file in os.listdir(queue_folder):
            if file.endswith(".mp3"):
                audio_name = file[:-4]
                self.analyze_track(audio_name, f"{queue_folder}/{file}")
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
            print(f"AUTOANALYSIS: Analyzing track {audio_name}")
            self.analysed.append(audio_name)
            self.analyze_track(audio_name, f"{folder}/{file}.mp3")

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

        self.analyze_track(self.queue[0], f"{folder_path}/{self.queue[0]}.mp3")

        threading.Thread(target=self.play_songs).start()

        # Analyze any songs that aren't ready
        self.concurrent_analyze(folder_path, self.queue[1:])
        print("Analyzed all songs")

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
