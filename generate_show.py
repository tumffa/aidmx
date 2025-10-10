import argparse
import json
from pathlib import Path
from src.services.queue_manager import QueueManager
from src.services.data_manager import DataManager
from src.services.qlc_manager import QLCManager

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(audio_name, delay, strobe=False, simple=False):
    config = load_config()
    setup_path = Path(config["setup_path"])
    setupfile_name = setup_path.stem
    data_manager = DataManager(config)
    qlc = QLCManager(setupfile_name, setup_path)
    queueservice = QueueManager(setupfile_name, data_manager, qlc)
    queueservice.analyze_file(audio_name, delay=delay, strobes=strobe, simple=simple)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a light show from an audio file.")
    parser.add_argument("audio_name", help="Name of the audio file (without extension)")
    parser.add_argument("--strobe", "-s", action="store_true", help="Enable strobe effects")
    parser.add_argument("--simple", "-m", action="store_true", help="Enable simple mode")
    parser.add_argument("--delay", "-d", type=int, default=500, help="Delay in ms before show start to allow for i.e. song track command to begin (default: 600ms)")
    args = parser.parse_args()
    main(args.audio_name, strobe=args.strobe, simple=args.simple, delay=args.delay)