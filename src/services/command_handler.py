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
        print("analyze <audio_name> [--strobe] [--simple] [--ola_delay=0.25)] [--qlc_delay=1)] [--qlc_lag=0.8955)]")
        print("-Delays are used to delay light show to match song playback")
        print("-qlc_lag used to downscale dimmer wait times for lag, scale down/up if beat flashes are too slow/fast")
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
            ola_delay = None
            qlc_delay = None
            qlc_lag = None
            for arg in command[1:]:
                if arg.lower() in ['--strobe', '-s', 'y']:
                    strobe = True
                elif arg.lower() in ['--simple', '-m']:
                    simple = True
                elif arg.lower().startswith('--ola_delay='):
                    try:
                        ola_delay = float(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid ola_delay value")
                elif arg.lower().startswith('--qlc_delay='):
                    try:
                        qlc_delay = float(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid qlc_delay value")
                        return
                elif arg.lower().startswith('--qlc_lag='):
                    try:
                        lag_value = float(arg.split('=', 1)[1])
                        self.queuemanager.structurer.set_qlc_lag(lag_value)
                    except ValueError:
                        print("Invalid qlc_lag value")
                        return
                elif not name:
                    name = arg
            if not name:
                print("Usage: analyze <audio_name> [--strobe] [--simple] [--ola_delay=seconds] [--qlc_delay=milliseconds] [--qlc_lag=percentage]")
                return
            if ola_delay is None or ola_delay < 0:
                ola_delay = 0.25
            if qlc_delay is None or qlc_delay < 0:
                qlc_delay = 1
            if qlc_lag is None or qlc_lag <= 0:
                qlc_lag = 0.8955
            self.queuemanager.analyze_file(name, strobe, simple, ola_delay=ola_delay, qlc_delay=qlc_delay, qlc_lag=qlc_lag)

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
