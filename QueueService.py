from QLCService import QXWHandler
from DataService import DataManager
import StatisticsService
import ShowStructurer
import os

class QueueManager:
    def __init__(self, setupfile="Newsetup.qxw"):
        self.queue = []
        self.ready = []
        self.dm = DataManager()
        self.qxw = QXWHandler(setupfile)
        self.structurer = ShowStructurer.ShowStructurer(self.dm)
        
    def analyze_track(self, audio_name, filepath):
        self.dm.extract_data(audio_name, filepath)
        StatisticsService.segment(audio_name, self.dm.get_song(audio_name), ["drums", "other"])
        self.structurer.generate_show(audio_name, self.qxw)
        self.ready.append(audio_name)

    def analyze_queue(self, queue_folder):
        for file in os.listdir(queue_folder):
            if file.endswith(".mp3"):
                audio_name = file[:-4]
                self.analyze_track(audio_name, f"{queue_folder}/{file}")
                self.queue.append(audio_name)