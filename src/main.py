import json
from pathlib import Path
from services.command_handler import CommandHandler
from services.queue_manager import QueueManager
from services.data_manager import DataManager
from services.qlc_manager import QLCManager


def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    """Main function"""

    config = load_config()
    setup_path = Path(config["setup_path"])
    setupfile_name = setup_path.stem
    data_manager = DataManager(config)
    qlc = QLCManager(setupfile_name, setup_path)
    queueservice = QueueManager(setupfile_name, data_manager, qlc)
    handler = CommandHandler(queueservice)
    handler.start()

if __name__ == "__main__":
    main()