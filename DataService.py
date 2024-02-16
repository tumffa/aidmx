import os
from pathlib import Path
import allin1
import json
from allin1 import demix, helpers
from dataclasses import asdict
from typing import Union, List
import StatisticsService


def update_struct_data(name, params, indent=2):
    # Load the existing data
    with open(f'./struct/{name}.json', 'r') as f:
        struct_data = json.load(f)

    # Update the segments with the given parameters
    for dictionary in params:
        struct_data.update(dictionary)

    # Save the updated data
    save_json(struct_data, './struct', name, indent)

def save_json(
    results: Union[dict, List[dict]],
    out_dir: str,
    name = None,
    indent = 2
):
    if not isinstance(results, list):
        results = [results]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if name is not None:
            out_path = out_dir / f"{name}.json"
        else:
            out_path = out_dir / f"{result['path']}.json"
        result['path'] = str(result['path'])

        json_str = json.dumps(result, indent=indent)
        out_path.write_text(json_str)

def get_struct_data(name):
    with open(f'./struct/{name}.json', 'r') as f:
        struct_data = json.load(f)
    return struct_data

def get_data(name):
    data = {}
    data["struct"] = get_struct_data(name)

class DataManager:
    def __init__(self):
        self._songs = {}
        if not os.path.exists("songdata.json"):
            with open("songdata.json", "w") as f:
                f.write("{}")
        else:
            self._get_json_data()

        for file in os.listdir("./struct"):
            if file.endswith(".json"):
                name = file[:-5]
                if name not in self._songs:
                    self._songs[name] = {}
                    self._songs[name]["analyzed"] = f"./struct/{name}.json"
                    with open(f"./struct/{name}.json", "r") as f:
                        data = json.load(f)
                        if "path" in data:
                            self._songs[name]["file"] = data["path"]
                    self._songs[name]["demixed"] = f"./demix/htdemucs/{name}/"

    def _get_json_data(self):
        with open("songdata.json", "r") as f:
            data = json.load(f)
            self._songs = data

    def _save_json_data(self):
        with open("songdata.json", "w") as f:
            json.dump(self._songs, f)

    def __str__(self):
        return str(self._songs)
    
    def get_song(self, song_name):
        if song_name in self._songs:
            return self._songs[song_name]
        else:
            return None
    
    def extract_data(self, audio_name, filepath):
        print(audio_name)
        if audio_name not in self._songs:
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
                print("AAAAAAAAAAAAAAAA")
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
        data = get_struct_data(audio_name)
        if "rms" not in data or "total_rms" not in data:
            StatisticsService.initialize_rms(self._songs[audio_name], audio_name)

        self._save_json_data()

        update_struct_data(audio_name, [{"filepath": filepath}])