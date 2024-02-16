from QueueService import QueueManager

# queuemanager = QueueManager("Newsetup.qxw")
# # queuemanager.analyze_queue("/mnt/e/MESHUGGAAAH")
# name = "avatar"
# queuemanager.analyze_track(name, f"./songs/{name}.mp3")

class ServiceHandler:
    def __init__(self):
        self.queuemanager = None
        
    def start(self):
        self.queuemanager = QueueManager("Newsetup.qxw")
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
        print("play <audio_name> - Play a track")
        print("exit - Exit the program")

    def handle_command(self, command):
        command = command.split()
        if command[0] == "analyze":
            if len(command) == 3:
                self.queuemanager.analyze_track(command[1], command[2])
            elif len(command) == 2:
                self.queuemanager.analyze_track(command[1], None)
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
        elif command[0] == "info":
            self.info()
        else:
            print("Invalid command")

if __name__ == "__main__":
    handler = ServiceHandler()
    handler.start()