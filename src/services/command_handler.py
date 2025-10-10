class CommandHandler:
    def __init__(self, queuemanager=None):
        self.queuemanager = queuemanager
        
    def start(self):
        while True:
            print(self.info())
            command = input()
            if command == 'exit':
                break
            else:
                self.handle_command(command)

    def info(self):
        print("Commands:")
        print("analyze <audio_name> [--strobe] [--simple] [--delay=ms] - Analyze a single track. Use --simple if you, for instance, only have a couple lights")
        print("folder <queue_folder> - Analyze all tracks in a folder")
        print("merge <audio_name> <folder> - Merge shows in folder into single showfile playlist (can be out of sync slightly)")
        print("sync - run this if there are previously analyzed struct files")
        print("exit - Exit the program")

    def handle_command(self, command):
        command = command.split()
        if command[0] == "analyze":
            strobe = False
            simple = False
            name = None
            delay = None
            for arg in command[1:]:
                if arg.lower() in ['--strobe', '-s', 'y']:
                    strobe = True
                elif arg.lower() in ['--simple', '-m']:
                    simple = True
                elif arg.lower().startswith('--delay='):
                    try:
                        delay = int(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid delay value, must be an integer (milliseconds)")
                        return
                elif not name:
                    name = arg
            if not name:
                print("Usage: analyze <audio_name> [--strobe] [--simple] [--delay=milliseconds]")
                return
            self.queuemanager.analyze_file(name, strobe, simple, delay)

        elif command[0] == "sync":
            self.queuemanager.sync_with_struct()
        elif command[0] == "analyzedata":
            name = command[1]
            self.queuemanager.analyze_data(name)
        elif command[0] == "info":
            self.info()
        elif command[0] == "merge":
            name = command[1]
            folder = command[2]
            self.queuemanager.merge_shows(name, folder)
        elif command[0] == "folder":
            folder = command[1]
            self.queuemanager.concurrent_analyze(folder)
        else:
            print("Invalid command")
