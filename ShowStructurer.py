import DataService

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe = {"above": {"1":{"id": 0, "dimmer": 7, "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35},
                                         "pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4,},

                                   "2":{"id": 1, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                        ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4},

                                   "3":{"id": 2, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                        ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4},

                                    "4":{"id": 3, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                         ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4},

                                    "5":{"id": 4, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                         ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4},
        
                                    "6":{"id": 5, "dimmer": 7,  "shutter": 8, "shutters": {"open": 255, "closed": 0, "flash": (10, 74), "fastflash": 35}
                                         ,"pan": 0, "tilt": 2, "panrange": 360, "tiltrange": 220, "color": 5, "colors": {"red": 87, "yellow": 39, "green": 70, "pink": 50, "fastflash": 183}, "movespeed": 4}},
                        "flood": {"1":{"id": 6, "dimmer": 3}}
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
    
    def _blackout(self, mode, comment=""):
        return f"blackout:{mode} //{comment}"
    
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

    def test(self, name, interval=None, length=30000.0):
        result = {}
        light_queue = Queue()
        result["name"] = "light"
        light_queue.enqueue(0)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group1 = self.universe["above"]
        beatinterval = show.bpminterval
        time = length
        switch = 0
        while time > 1:
            temp = []
            for fixture in group1.values():
                if switch == 0:
                    if int(fixture["id"]) > 2:  # Every other fixture
                        line = self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}")
                        temp.append(line)
                    else:
                        line = self._setfixture(fixture["id"], fixture["dimmer"], 0, f"{time}")
                        temp.append(line)
                else:
                    if int(fixture["id"]) <= 2:
                        line = self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}")
                        temp.append(line)
                    else:
                        line = self._setfixture(fixture["id"], fixture["dimmer"], 0, f"{time}")
                        temp.append(line)
            switch = 1 - switch
            light_queue.enqueue(temp)
            wait = beatinterval*1000
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                light_queue.enqueue(wait)
        result["queue"] = light_queue
        return result

    def test2(self, name, length=30000.0):
        result = {}
        spin_queue = Queue()
        result["name"] = "spin"
        spin_queue.enqueue(0)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group1 = self.universe["above"]
        time = length
        start_tilt = 23
        start_pan = 126
        fixture = group1["1"]
        pan_offset = 35
        tilt_offset = int(pan_offset * (fixture["tiltrange"]/fixture["panrange"])) + 10
        print(tilt_offset)
        circle_time = 2 * show.beatinterval
        panspeed = self.calculate_pan_speed(fixture, pan_offset, circle_time/4)
        print(panspeed)
        for fixture in group1.values():
            if int(fixture["id"]) % 2 == 0:
                color = "yellow"
                shutter = "open"
            else:
                shutter = "open"
                color = "pink"
            self._write(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"][shutter], "Open shutters"))
            self._write(self._setfixture(fixture["id"], fixture["color"], fixture["colors"][color], "Set color to fastflash"))
            #turn dimmer to full
            self._write(self._setfixture(fixture["id"], fixture["dimmer"], 255, "Turn dimmer to full"))
            self._write(self._setfixture(fixture["id"], fixture["pan"], 126, "Turn fixture 180 degrees"))
            self._write(self._setfixture(fixture["id"], fixture["tilt"], 0, "Set tilt to 23"))
            self._write(self._setfixture(fixture["id"], fixture["movespeed"], panspeed, "Set move speed"))
        pan = -1
        tilt = 0
        pans = {"0": 126, "1": 126, "2": 126, "3": 126, "4": 126, "5": 126}
        tilts = {"0": 30, "1": 30, "2": 30, "3": 30, "4": 30, "5": 30}
        while time > 1:
            temp = []
            for fixture in group1.values():
                current_pan = pans[str(fixture["id"])]
                current_tilt = tilts[str(fixture["id"])]
                if int(fixture["id"]) % 2 == 0:
                    pan_offset2 = -pan_offset
                    tilt_offset2 = -tilt_offset
                else:
                    pan_offset2 = pan_offset
                    tilt_offset2 = tilt_offset
                if pan >= 0:
                    line = self._setfixture(fixture["id"], fixture["pan"], current_pan + pan_offset2, f"{time}")
                    temp.append(line)
                    pans[fixture["id"]] = current_pan + pan_offset
                if tilt >= 0:
                    line = self._setfixture(fixture["id"], fixture["tilt"], current_tilt + tilt_offset2, f"{time}")
                    temp.append(line)
                    tilts[fixture["id"]] = current_tilt + tilt_offset
                if pan < 0:
                    line = self._setfixture(fixture["id"], fixture["pan"], current_pan - pan_offset2, f"{time}")
                    temp.append(line)
                    pans[fixture["id"]] = current_pan - pan_offset
                if tilt < 0:
                    line = self._setfixture(fixture["id"], fixture["tilt"], current_tilt - tilt_offset2, f"{time}")
                    temp.append(line)
                    tilts[fixture["id"]] = current_tilt - tilt_offset
            pan += 1
            tilt += 1
            if pan >= 2:
                pan = -2
            if tilt >= 2:
                tilt = -2
            spin_queue.enqueue(temp)
            wait =  (circle_time/4)*1000
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                spin_queue.enqueue(wait)
        result["queue"] = spin_queue
        return result

    def test3(self, name, interval=None, length=30000.0):
        result = {}
        flood_queue = Queue()
        result["name"] = "flood"
        flood_queue.enqueue(0)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group = self.universe["flood"]
        time = length
        if interval:
            beatinterval = interval
        else:
            beatinterval = show.bpminterval
        print(beatinterval)
        lightvalue = 255
        wait = beatinterval * 1000  # Adjust wait to be equal to the beat interval in milliseconds
        #make the flood light flash and go down to 0 every beat
        while time > 1:
            temp = []
            if lightvalue <= 0:
                lightvalue = 255
                line = self._setfixture(group["1"]["id"], group["1"]["dimmer"], lightvalue, f"{time}")
                temp.append(line)
            else:
                lightvalue -= 255/(1000*beatinterval/wait)
                print(lightvalue)
                line = self._setfixture(group["1"]["id"], group["1"]["dimmer"], lightvalue, f"{time}")
                temp.append(line)
            flood_queue.enqueue(temp)
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                flood_queue.enqueue(wait)
        result["queue"] = flood_queue
        return result
    
    def pause(self, length, type="blackout"):
        result = {}
        result["name"] = "pause"
        if type == "blackout":
            blackout_queue = Queue()
            blackout_queue.enqueue(0)
            temp = []
            temp.append(self._blackout("on", f"Blackout for {length} seconds"))
            blackout_queue.enqueue(temp)
            wait = length * 1000
            blackout_queue.enqueue(wait)
            temp = []
            temp.append(self._blackout("off", f"Blackout off"))
            blackout_queue.enqueue(temp)
            result["queue"] = blackout_queue
        return result

    def combine(self, queues):
        command_queues = {}
        for queue in queues:
            command_queues[queue["name"]] = queue["queue"]
        times = {}
        for queue in queues:
            times[queue["name"]] = queue["queue"].dequeue()

        while len(times) > 0:
            print(times)
            min_time = min(times.values())
            min_queues = [k for k, v in times.items() if v == min_time]
            if 'flood' in min_queues:
                q = ('flood', min_time)
            else:
                q = min_queues[0], min_time

            queue = command_queues[q[0]]
            self._write(self._wait(q[1]))
            for name in times:
                times[name] -= q[1]
            commands = queue.dequeue()
            if commands == None:
                del times[q[0]]
                del command_queues[q[0]]
                continue
            for command in commands:
                self._write(command)
            wait = queue.dequeue()
            if wait:
                times[q[0]] = wait

    def generate_segment(self, name):
        time = 15000.0
        pauses = [(6, 8)]
        queues = []
        queues.append(self.test(name, length=13000))
        queues.append(self.test2(name, length=13000))
        queues.append(self.test3(name, length=13000))
        self.combine(queues)
        queues = []
        queues.append(self.pause((8-6)))
        self.combine(queues)
        queues = []
        queues.append(self.test(name, length=7000))
        queues.append(self.test2(name, length=7000))
        queues.append(self.test3(name, length=7000))
        self.combine(queues)

class Queue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, item):
        self.queue.append(item)

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            return None

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            return None

    def edit(self, index, item):
        if index < len(self.queue):
            self.queue[index] = item
        else:
            print("Index out of range")

class Show:
    def __init__(self, name, struct, song_data):
        self.name = name
        self.struct = struct
        self.song_data = song_data
        self.mp3_path = song_data["file"]
        self.bpminterval = 60 / (struct["bpm"]*1.04)
        self.beatinterval = 60 / struct["bpm"]
