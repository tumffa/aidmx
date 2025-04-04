import os
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
