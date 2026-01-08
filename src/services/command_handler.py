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
        print("\n\nCommands:")
        print("\nanalyze <audio_name> [-st] [-si] [-qd=SEC)] [-ql=FRAC)]")
        print("--Starts analysis from scratch and generates show for <audio_name> file in songs folder")
        print("-- Strobes can be enabled with -st")
        print("--Delay -qd is used to delay qlc light show to match powershell command song playback")
        print("--Fraction -ql is used to downscale dimmer wait times for lag, scale down/up if beat flashes are too slow/fast")
        print("\ngenerate <audio_name> - Generate show with existing struct data for <audio_name>")
        print("--use 'sync' command first to sync struct data")
        print("\nola <audio_name> [-od=SEC] [-u=N] [-s=SEC]")
        print("--Play OLA show for <audio_name> with existing struct data with delay -od, universe -u, and start time -s")
        print("\nfolder <queue_folder> - Analyze all tracks in a folder")
        print("\nmerge <audio_name> <folder> - Merge shows in folder into single showfile playlist (can be out of sync slightly)")
        print("\nsync - run this if there are previously analyzed struct files")
        print("\nexit - Exit the program")

    def handle_command(self, command):
        command = command.split()
        if not command:
            return

        if command[0] in ("analyze", "generate"):
            strobe = False
            simple = False
            name = None
            qlc_delay = 1.0
            qlc_lag = 0.8955

            for arg in command[1:]:
                if arg.lower() in ['--strobe', '-st', 'y']:
                    strobe = True
                elif arg.lower() in ['--simple', '-si']:
                    simple = True
                elif arg.lower() in ['--qlc_delay', '-qd']:
                    try:
                        qlc_delay = float(arg.split('=')[1])
                    except ValueError:
                        print("Invalid qlc_delay value")
                        return
                elif arg.lower() in ['--qlc_lag', '-ql']:
                    try:
                        qlc_lag = float(arg.split('=')[1])
                    except ValueError:
                        print("Invalid qlc_lag value")
                        return

                elif not name:
                    name = arg

            if not name:
                print("Usage: analyze/generate <audio_name> [-st] [-si] [-qd=SEC] [-ql=FRAC]")
                return

            if command[0] == "analyze":
                self.queuemanager.analyze_file(
                    name,
                    strobe,
                    simple,
                    qlc_delay=qlc_delay,
                    qlc_lag=qlc_lag,
                )
            else:
                self.queuemanager.generate(
                    name,
                    strobe,
                    simple,
                    qlc_delay=qlc_delay,
                    qlc_lag=qlc_lag,
                )

        elif command[0] == "ola":
            name = None
            ola_delay = None
            qlc_delay = None
            qlc_lag = None
            universe = None
            start = 0.0

            for arg in command[1:]:
                al = arg.lower()
                if al.startswith('-od='):
                    try:
                        ola_delay = float(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid ola_delay value")
                        return
                elif al.startswith('-u='):
                    try:
                        universe = int(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid universe value")
                        return
                elif al.startswith('-s='):
                    try:
                        start = float(arg.split('=', 1)[1])
                    except ValueError:
                        print("Invalid start time value")
                        return
                elif not name:
                    name = arg

            if not name:
                print("Usage: ola <audio_name> [-od=SEC] [-u=N] [-s=SEC]")
                return

            if ola_delay is None or ola_delay < 0:
                ola_delay = 0.04
            if qlc_delay is None or qlc_delay < 0:
                qlc_delay = 1.0
            if qlc_lag is None or qlc_lag <= 0:
                qlc_lag = 0.8955
            if universe is None or universe <= 0:
                universe = 1

            # Play OLA show with given delay/universe
            self.queuemanager.play_ola_show(name, delay=ola_delay, universe=universe)

        elif command[0] == "sync":
            self.queuemanager.sync_with_struct()
        elif command[0] == "generate":
            name = command[1]
            self.queuemanager.generate(name)
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