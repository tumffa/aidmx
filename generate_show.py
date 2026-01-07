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

def main(audio_name, strobe=False, simple=False, qlc_delay=1, qlc_lag=0.8955):
    config = load_config()
    setup_path = Path(config["setup_path"])
    setupfile_name = setup_path.stem
    data_manager = DataManager(config)
    qlc = QLCManager(setupfile_name, setup_path)
    queueservice = QueueManager(setupfile_name, data_manager, qlc)
    queueservice.analyze_file(audio_name, strobes=strobe, simple=simple, qlc_delay=qlc_delay, qlc_lag=qlc_lag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a light show from an audio file.")
    parser.add_argument("audio_name", help="Name of the audio file (without extension)")
    parser.add_argument("-st", "-s", action="store_true", help="Enable strobe effects")
    parser.add_argument("-si", "-m", action="store_true", help="Enable simple mode")
    parser.add_argument("-d", "-d", type=float, default=1, help="Delay in seconds before show start to allow for i.e. song track command to begin (default: 1 sec)")
    parser.add_argument("-l", "-l", type=float, default=0.8955, help="Scaling factor for wait times to compensate for QLC+ lag (default: 0.8955)")
    args = parser.parse_args()
    main(args.audio_name, strobe=args.st, simple=args.si, qlc_delay=args.d, qlc_lag=args.l)