import json
from pathlib import Path
from services.commandhandler import CommandHandler
from services.queue_service import QueueManager
from services.dataservice import DataManager
from services.qlc_service import QLCHandler


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
    qlc = QLCHandler(setupfile_name, setup_path)
    queueservice = QueueManager(setupfile_name, data_manager, qlc)
    handler = CommandHandler(queueservice)
    handler.start()

if __name__ == "__main__":
    main()