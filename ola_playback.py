import argparse
import json
from pathlib import Path
from src.services.queue_manager import QueueManager
from src.services.data_manager import DataManager

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(audio_name, delay=1, universe=1):
    config = load_config()
    setup_path = Path(config["setup_path"])
    setupfile_name = setup_path.stem
    data_manager = DataManager(config)
    queueservice = QueueManager(setupfile_name, data_manager)
    queueservice.play_ola_show(audio_name, delay, universe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play OLA DMX show and audio file.")
    parser.add_argument("audio_name", help="Name of the audio file (without extension)")
    parser.add_argument("-d", "--delay", type=float, default=0.06, help="Delay in seconds before DMX playback (default: 0.06 sec)")
    parser.add_argument("-u", "--universe", type=int, default=1, help="OLA DMX universe to use (default: 1)")
    args = parser.parse_args()
    main(args.audio_name, delay=args.delay, universe=args.universe)