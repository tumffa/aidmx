import os
from pathlib import Path
import allin1
from allin1 import demix, helpers

class DataManager:
    def __init__(self):
        self._songs = {}

    def __str__(self):
        return str(self._songs)
    
    def get_song(self, song_name):
        return self._songs[song_name]
    
    def extract_data(self, audio_name, filepath):
        print(audio_name)
        if audio_name in self._songs:
            return "Song has already been analyzed"
        self._songs[audio_name] = {}
        self._songs[audio_name]["file"] = filepath
            
        # Run the demix function
        path = f"./struct/{audio_name}.json"
        try:
            if self._songs[audio_name]["analyzed"] != path:
                self._songs[audio_name]["analyzed"] = path
        except KeyError:
            self._songs[audio_name]["analyzed"] = path
        try:
            if not os.path.exists(path):
                analyzed = allin1.analyze(Path(filepath))
                helpers.save_results(analyzed, './struct', audio_name)
            else:
                print("Already analyzed")
        except Exception as e:
            print(f"Error while analyzing: {e}")
        path = f"./demix/htdemucs/{audio_name}/"
        try:
            if self._songs[audio_name]["demixed"] != path:
                self._songs[audio_name]["demixed"] = path
        except KeyError:
            self._songs[audio_name]["demixed"] = path
        try:
            if not os.path.exists(f"./demix/htdemucs/{audio_name}"):
                demix.demix([Path(filepath)], demix_dir=Path('./demix'), device='cuda:0')
            else:
                print("Already demixed")
        except Exception as e:
            print(f"Error while demixing: {e}")
        categories = ['vocals']
        for category in categories:
            print("Category: " + category)
            try:
                if not os.path.exists(f"./struct/{audio_name}{category}.json"):
                    # Construct the path to the JSON file
                    analyzed = allin1.analyze(Path(f"./demix/htdemucs/{audio_name}/{category}.wav"))
                    helpers.save_results(analyzed, './struct', f"{audio_name}{category}")
            except Exception as e:
                print(f"Json for {category} already exists")
