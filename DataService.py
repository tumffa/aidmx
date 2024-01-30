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
        # List of categories
        categories = ['drums', 'bass', 'vocals', 'other']
        # For each category
        for category in categories:
            try:
                if not os.path.exists(f"./struct/{audio_name}{category}.json"):
                    # Construct the path to the JSON file
                    json_path = os.path.join('./struct', f'{audio_name}{category}.json')

                    # Check if the JSON file exists
                    if not os.path.exists(json_path):
                        pass  # Add your code here
            except Exception as e:
                print(f"Error while processing category {category}: {e}")
