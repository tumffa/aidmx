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
            if len(command) == 3:
                self.queuemanager.analyze_file(command[1], command[2])
            elif len(command) == 2:
                self.queuemanager.analyze_file(command[1], None)
            else:
                print("Invalid number of arguments")
        elif command[0] == "af":
            if len(command) == 2:
                self.queuemanager.choose_folder(command[1])
            else:
                print("Invalid number of arguments")
        elif command[0] == "play":
            if len(command) == 2:
                self.queuemanager.play_track(command[1])
            else:
                print("Invalid number of arguments")
        elif command[0] == "a":
            if len(command) == 2:
                self.queuemanager.auto_play_track(command[1])
            else:
                print("Invalid number of arguments")
        elif command[0] == "sync":
            self.queuemanager.sync_with_struct()
        elif command[0] == "analyzedata":
            name = command[1]
            self.queuemanager.analyze_data(name)
        elif command[0] == "info":
            self.info()
        else:
            print("Invalid command")
