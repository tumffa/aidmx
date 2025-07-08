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
        print("analyze <audio_name> <filepath> - Analyze a track")
        print("analyze_queue <queue_folder> - Analyze all tracks in a folder")
        print("analyzedata <audio_name> - generate from analyzed file")
        print("play <audio_name> - Play a track")
        print("sync - load sons from struct")
        print("exit - Exit the program")

    def handle_command(self, command):
        command = command.split()
        if command[0] == "analyze":
            strobe = True
            if len(command) == 3:
                strobe = False
            self.queuemanager.analyze_file(command[1], None, strobe)
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
