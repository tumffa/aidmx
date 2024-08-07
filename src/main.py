import os
from services.commandhandler import CommandHandler
from services.queue_service import QueueManager
from services.dataservice import DataManager
from services.qlc_service import QLCHandler


def main():
    """Main function"""

    current_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_path, os.pardir))  # Navigate up one directory
    setupfile = "Newsetup.qxw"
    data_manager = DataManager(root_path)
    qlc = QLCHandler(setupfile, root_path)
    queueservice = QueueManager(setupfile, data_manager, qlc)
    handler = CommandHandler(queueservice)
    handler.start()

if __name__ == "__main__":
    main()