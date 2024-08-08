from pathlib import Path
import allin1
import json
from allin1 import demix, helpers
from dataclasses import asdict
from typing import Union, List
from services import audio_analyzer


class DataManager:
    def __init__(self, config):
        self.data_path = Path(config["data_path"])
        self.struct_path = Path(config["struct_path"])
        self.demix_path = Path(config["demix_path"])
        self.songs = {}

    def _get_json_data(self):
        with open(self.data_path / "songdata.json", "r") as f:
            data = json.load(f)
            self.songs = data

    def _save_json_data(self):
        with open(self.data_path / "songdata.json", "w") as f:
            json.dump(self.songs, f)

    def __str__(self):
        return str(self.songs)
    
    def get_song(self, song_name):
        return self.songs.get(song_name, None)
    
    def extract_data(self, audio_name, filepath):
        print(audio_name)
        if audio_name not in self.songs:
            self.songs[audio_name] = {}
            self.songs[audio_name]["file"] = filepath

        # Run the demix function
        path = self.struct_path / f"{audio_name}.json"
        try:
            if self.songs[audio_name].get("analyzed") != str(path):
                self.songs[audio_name]["analyzed"] = str(path)
        except KeyError:
            self.songs[audio_name]["analyzed"] = str(path)
        try:
            if not path.exists():
                analyzed = allin1.analyze(Path(filepath))
                helpers.save_results(analyzed, self.struct_path, audio_name)
            else:
                print("Already analyzed")
        except Exception as e:
            print(f"Error while analyzing: {e}")
        
        path = self.demix_path / "htdemucs" / audio_name
        try:
            if self.songs[audio_name].get("demixed") != str(path):
                self.songs[audio_name]["demixed"] = str(path)
        except KeyError:
            self.songs[audio_name]["demixed"] = str(path)
        try:
            if not path.exists():
                demix.demix([Path(filepath)], demix_dir=self.demix_path, device='cuda:0')
            else:
                print("Already demixed")
        except Exception as e:
            print(f"Error while demixing: {e}")

        data = self.get_struct_data(audio_name)
        if "rms" not in data or "total_rms" not in data:
            segments = self.get_struct_data_by_key(audio_name, "segments")
            params = audio_analyzer.initialize_rms(self.songs[audio_name], 
                                                   audio_name, 
                                                   self.demix_path, 
                                                   segments)
            self.update_struct_data(audio_name, params, indent=2)
            struct_data = self.get_struct_data(audio_name)
            params = audio_analyzer.struct_stats(self.songs[audio_name], 
                                                       audio_name, 
                                                       struct_data=struct_data)
            self.update_struct_data(audio_name, params, indent=2)

        self._save_json_data()

    def update_struct_data(self, name, params, indent=2):
        # Load the existing data
        with open(self.struct_path / f"{name}.json", 'r') as f:
            struct_data = json.load(f)

        # Update the segments with the given parameters
        for dictionary in params:
            struct_data.update(dictionary)

        # Save the updated data
        self.save_json(struct_data, self.struct_path, name, indent)

    def save_json(self,
        results: Union[dict, List[dict]],
        out_dir: Path,
        name = None,
        indent = 2
        ):
        
        if not isinstance(results, list):
            results = [results]
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            if name is not None:
                out_path = out_dir / f"{name}.json"
            else:
                out_path = out_dir / f"{result['path']}.json"
            result['path'] = str(result['path'])

            json_str = json.dumps(result, indent=indent)
            out_path.write_text(json_str)

    def get_struct_data(self, name):
        with open(self.struct_path / f"{name}.json", 'r') as f:
            struct_data = json.load(f)
        return struct_data
    
    def get_struct_data_by_key(self, name, key):
        with open(self.struct_path / f"{name}.json", 'r') as f:
            struct_data = json.load(f)
        return struct_data.get(key)

    def get_data(self, name):
        data = {}
        data["struct"] = self.get_struct_data(name)
        return data
