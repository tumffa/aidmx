class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe = {"above": {"1":{"id": 0, "dimmer": 5, "pan": 0, "tilt": 1, "color": 7, "colors": {"red": 11, "yellow": 18, "green": 27}, "movespeed": 4},
                                   "2":{"id": 1, "dimmer": 5, "pan": 0, "tilt": 1, "color": 7, "colors": {"red": 11, "yellow": 18, "green": 27}, "movespeed": 4},
                                   "3":{"id": 2, "dimmer": 5, "pan": 0, "tilt": 1, "color": 7, "colors": {"red": 11, "yellow": 18, "green": 27}, "movespeed": 4},
                                    "4":{"id": 3, "dimmer": 5, "pan": 0, "tilt": 1, "color": 7, "colors": {"red": 11, "yellow": 18, "green": 27}, "movespeed": 4}}
                                }

    def get_songdata(self, name):
        struct = self.dm.get_struct_data(name)
        song_data = self.dm.get_song(name)
        return struct, song_data
    
    def create_show(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show

    def _setfixture(self, fixture, channel, value, comment=""):
        return f"setfixture:{fixture} ch:{channel} val:{value} //{comment}"

    def _wait(self, time, comment=""):
        return f"wait:{time} //{comment}"

class Show:
    def __init__(self, name, struct, song_data):
        self.name = name
        self.struct = struct
        self.song_data = song_data
        self.mp3_path = song_data["file"]