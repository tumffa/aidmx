import DataService

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe = {"above": {"1":{"id": 0, "dimmer": 7, "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35},
                                         "pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 11, "yellow": 18, "green": 27, "fastflash": 183}, "movespeed": 4,},

                                   "2":{"id": 1, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                        ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 11, "yellow": 18, "green": 27, "fastflash": 183}, "movespeed": 4},

                                   "3":{"id": 2, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                        ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 11, "yellow": 18, "green": 27, "fastflash": 183}, "movespeed": 4},

                                    "4":{"id": 3, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                         ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 11, "yellow": 18, "green": 27, "fastflash": 183}, "movespeed": 4}}
                                }
        self.file = "showfile.txt"
        #empty the showfile
        with open(self.file, "w") as f:
            f.write("")

    def get_songdata(self, name):
        struct = DataService.get_struct_data(name)
        song_data = self.dm.get_song(name)
        return struct, song_data
    
    def create_show(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show

    def _setfixture(self, fixture, channel, value, comment=""):
        if value > 255:
            value = 255
        if value < 0:
            value = 0
        return f"setfixture:{fixture} ch:{channel} val:{value} //{comment}"

    def _wait(self, time, comment=""):
        return f"wait:{int(time)} //{comment}"
    
    def _write(self, line):
        with open(self.file, "a") as f:
            f.write(line + "\n")

    def calculate_pan_speed(self, fixture, angle, time):
        range = fixture["panrange"]
        rate = 255 / range
        dmx_angle = angle * rate
        return dmx_angle / time
    
    def calculate_tilt_speed(self, fixture, angle, time):
        range = fixture["tiltrange"]
        rate = 255 / range
        dmx_angle = angle * rate
        return dmx_angle / time

    def test(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group1 = self.universe["above"]
        time = 30000.0
        color_state = 0
        switch = 0
        for fixture in group1.values():
            self._write(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["fastflash"], "Open shutters"))
            self._write(self._setfixture(fixture["id"], fixture["color"], fixture["colors"]["fastflash"], "Set color to fastflash"))
            #turn fixtures 180degrees
            self._write(self._setfixture(fixture["id"], fixture["pan"], 126, "Turn fixture 180 degrees"))
        while time > 0:
            for fixture in group1.values():
                if switch == 0:
                    if int(fixture["id"]) % 2 == 0:  # Every other fixture
                        self._write(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}"))
                    else:
                        self._write(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"{time}"))
                else:
                    if int(fixture["id"]) % 2 != 0:
                        self._write(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}"))
                    else:
                        self._write(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"{time}"))
            switch = 1 - switch
            self._write(self._wait(show.beatinterval, f"{time}"))
            time -= show.beatinterval*1000

    def test2(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group1 = self.universe["above"]
        time = 15000.0
        start_tilt = 23
        start_pan = 126
        fixture = group1["1"]
        pan_offset = 35
        tilt_offset = int(pan_offset * (fixture["tiltrange"]/fixture["panrange"]))
        print(tilt_offset)
        circle_time = 1
        panspeed = self.calculate_pan_speed(fixture, pan_offset, circle_time/4)
        print(panspeed)
        for fixture in group1.values():
            self._write(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["fastflash"], "Open shutters"))
            self._write(self._setfixture(fixture["id"], fixture["color"], fixture["colors"]["fastflash"], "Set color to fastflash"))
            #turn dimmer to full
            self._write(self._setfixture(fixture["id"], fixture["dimmer"], 255, "Turn dimmer to full"))
            self._write(self._setfixture(fixture["id"], fixture["pan"], 126, "Turn fixture 180 degrees"))
            self._write(self._setfixture(fixture["id"], fixture["tilt"], 0, "Set tilt to 23"))
            self._write(self._setfixture(fixture["id"], fixture["movespeed"], panspeed, "Set move speed"))
        pan = -1
        tilt = 0
        pans = {"0": 126, "1": 126, "2": 126, "3": 126}
        tilts = {"0": 23, "1": 23, "2": 23, "3": 23}
        while time > 0:
            for fixture in group1.values():
                current_pan = pans[str(fixture["id"])]
                current_tilt = tilts[str(fixture["id"])]
                if pan >= 0:
                    self._write(self._setfixture(fixture["id"], fixture["pan"], current_pan + pan_offset, f"{time}"))
                    pans[fixture["id"]] = current_pan + pan_offset
                if tilt >= 0:
                    self._write(self._setfixture(fixture["id"], fixture["tilt"], current_tilt + tilt_offset, f"{time}"))
                    tilts[fixture["id"]] = current_tilt + tilt_offset
                if pan < 0:
                    self._write(self._setfixture(fixture["id"], fixture["pan"], current_pan - pan_offset, f"{time}"))
                    pans[fixture["id"]] = current_pan - pan_offset
                if tilt < 0:
                    self._write(self._setfixture(fixture["id"], fixture["tilt"], current_tilt - tilt_offset, f"{time}"))
                    tilts[fixture["id"]] = current_tilt - tilt_offset
            pan += 1
            tilt += 1
            if pan >= 2:
                pan = -2
            if tilt >= 2:
                tilt = -2
            self._write(self._wait((circle_time/4)*1000, f"{time}"))
            time -= (circle_time/4)*1000
                    


class Show:
    def __init__(self, name, struct, song_data):
        self.name = name
        self.struct = struct
        self.song_data = song_data
        self.mp3_path = song_data["file"]
        self.beatinterval = 60 / struct["bpm"]
