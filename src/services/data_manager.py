import allin1
import os
import json
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from typing import Union, List
from src.services import audio_analysis
from dataclasses import asdict, is_dataclass


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
    
    def return_path(self, name):
        # Return absolute data, struct, or demix path
        paths = {
            "data": self.data_path.resolve(),
            "struct": self.struct_path.resolve(),
            "demix": self.demix_path.resolve()
        }
        return paths.get(name)
    
    def get_song(self, song_name):
        return self.songs.get(song_name, None)
    
    def convert_to_wav(self, filepath):
        # Convert the file to wav
        audio = AudioSegment.from_file(filepath)
        audio.export(filepath[:-4] + ".wav", format="wav")
        os.remove(filepath)
        return filepath[:-4] + ".wav"
    
    def extract_data(self, audio_name, filepath):
        print(f"--Extracting data for {audio_name}")
        # Check if the file is an mp3 and convert it to wav
        if filepath.endswith(".mp3"):
            filepath = self.convert_to_wav(filepath)
        if audio_name not in self.songs:
            self.songs[audio_name] = {}
            self.songs[audio_name]["file"] = filepath

        os.environ["MKL_THREADING_LAYER"] = "GNU"

        # Run the demix function
        path = self.struct_path / f"{audio_name}.json"
        try:
            if self.songs[audio_name].get("analyzed") != str(path):
                self.songs[audio_name]["analyzed"] = str(path)
        except KeyError:
            self.songs[audio_name]["analyzed"] = str(path)
        try:
            if not path.exists():
                print(f"----Running through Allin1")
                analyzed = allin1.analyze(Path(filepath))
                self.save_results(analyzed, self.struct_path, audio_name)
            else:
                print("----Allin1 results already exist")
        except Exception as e:
            print(f"----Error while running Allin1: {e}")
        
        path = self.demix_path / "htdemucs" / audio_name
        # true if drums, vocals, bass, other.wav in audio_name
        demix_exists = all(path.exists() for path in [path / f"{inst}.wav" for inst in ["drums", "vocals", "bass", "other"]])
        try:
            if self.songs[audio_name].get("demixed") != str(path):
                self.songs[audio_name]["demixed"] = str(path)
        except KeyError:
            self.songs[audio_name]["demixed"] = str(path)
        try:
            if not demix_exists:
                print(f"----Demixed audio not found, demixing with Allin1.demix")
                allin1.demix.demix([Path(filepath)], demix_dir=self.demix_path, device='cuda:0')
            else:
                print("----Already demixed")
        except Exception as e:
            print(f"----Error while demixing: {e}")

        data = self.get_struct_data(audio_name)
        if "rms" not in data or "total_rms" or "larsnet_drums_y" not in data:
            segments = self.get_struct_data_by_key(audio_name, "segments")
            struct_data = self.get_struct_data(audio_name)
            params = audio_analysis.initialize_song_metrics(self.songs[audio_name], 
                                                    struct_data=struct_data)
            self.update_struct_data(audio_name, params, indent=2)
            struct_data = self.get_struct_data(audio_name)
            params = audio_analysis.struct_stats(self.songs[audio_name], 
                                                       audio_name, 
                                                       struct_data=struct_data)
            self.update_struct_data(audio_name, params, indent=2)

        self._save_json_data()
        print(f"--Data extraction complete for {audio_name}")

    def sync_with_struct(self):
        for file in self.struct_path.resolve().iterdir():
            if file.suffix == '.json':
                song_name = file.stem
                if song_name not in self.songs:
                    self.update_songdata(song_name)
    
    def update_songdata(self, song_name):
        content = {
            "file": f"{self.data_path.resolve()}/songs/{song_name}.wav",
            "analyzed": f"{self.struct_path.resolve()}/{song_name}.json",
            "demixed": f"{self.demix_path.resolve()}/htdemucs/{song_name}"
            }
        self.songs[song_name] = content
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
    
    def get_song_length(self, name):
        # Get the length of the song in milliseconds
        song = self.get_song(name)
        if not song:
            print(f"Song {name} not found in data.")
            return 0
        filepath = song["file"]
        audio = AudioSegment.from_file(filepath)
        return len(audio)

    def get_data(self, name):
        data = {}
        data["struct"] = self.get_struct_data(name)
        return data
    
    def save_results(self,
        results: Union[dict, List[dict]],
        out_dir: Path,
        name: str,
        indent: int = 2
        ):
        """
        Save AllIn1 analysis results to out_dir/<name>.json and associated
        activations/embeddings files if present.
        Modified version of allin1.helpers.save_results
        """
        if not isinstance(results, list):
            results = [results]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            try:
                if is_dataclass(result):
                    result_dict = asdict(result)
                elif isinstance(result, dict):
                    result_dict = dict(result)
                else:
                    result_dict = dict(result.__dict__) if hasattr(result, "__dict__") else dict(result)
            except Exception:
                result_dict = {}

            activations = None
            embeddings = None
            if "activations" in result_dict:
                activations = result_dict.pop("activations")
            elif hasattr(result, "activations"):
                activations = getattr(result, "activations", None)

            if "embeddings" in result_dict:
                embeddings = result_dict.pop("embeddings")
            elif hasattr(result, "embeddings"):
                embeddings = getattr(result, "embeddings", None)

            if activations is not None:
                try:
                    np.savez(str(out_dir / f"{name}.activ.npz"), **activations)
                except Exception:
                    pass

            if embeddings is not None:
                try:
                    np.save(str(out_dir / f"{name}.embed.npy"), embeddings)
                except Exception:
                    pass

            result_dict["path"] = str(name)
            json_str = json.dumps(result_dict, indent=indent)
            (out_dir / f"{name}.json").write_text(json_str)
