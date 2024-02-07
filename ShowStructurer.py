import DataService
import re

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
        return show

    def _setfixture(self, fixture, channel, value, comment=""):
        if value > 255:
            value = 255
        if value < 0:
            value = 0
        return f"setfixture:{fixture} ch:{channel} val:{value} //{comment}"

    def _wait(self, time, comment=""):
        if time < 0:
            time = 0
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

    def alternate(self, name, show, length=30000.0, start=0, queuename="alternate"):
        result = {}
        light_queue = Queue()
        result["name"] = queuename
        light_queue.enqueue(start)
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

    def spin(self, name, show, length=30000.0, start=0, queuename="spin"):
        result = {}
        spin_queue = Queue()
        result["name"] = queuename
        spin_queue.enqueue(start)
        self.shows[name] = show
        group1 = self.universe["above"]
        time = length
        start_tilt = 23
        start_pan = 126
        fixture = group1["1"]
        pan_offset = 35
        tilt_offset = int(pan_offset * (fixture["tiltrange"]/fixture["panrange"])) + 10
        print(tilt_offset)
        circle_time = 2 * (60/128)
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
    
    def swing(self, name, show, length=30000.0, start=0, queuename="swing"):
        result = {}
        swing_queue = Queue()
        result["name"] = queuename
        swing_queue.enqueue(start)
        group = self.universe["above"]
        time = length
        beatinterval = show.bpminterval
        tiltspeed = self.calculate_tilt_speed(group["1"], 80, beatinterval)
        while time > 1:
            temp = []
            switch = 0
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["movespeed"], tiltspeed, f"Set move speed"))
                if switch == 0:
                    if int(fixture["id"]) % 2 == 0:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 80, f"Tilt to 80"))
                    else:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 0, f"Tilt to 80"))
                else:
                    if int(fixture["id"]) % 2 == 0:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 0, f"Tilt to 0"))
                    else:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 80, f"Tilt to 0"))
            switch = 1 - switch
            swing_queue.enqueue(temp)
            wait = beatinterval*1000
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                swing_queue.enqueue(wait)
        result["queue"] = swing_queue
        return result

    def flood(self, name, interval=None, length=30000.0, start=0, queuename="flood"):
        result = {}
        flood_queue = Queue()
        result["name"] = queuename
        flood_queue.enqueue(start)
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
        lightvalue = 0
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
    
    def pulse(self, name, show, dimmer=140, length=30000.0, start=0, queuename="pulse"):
        result = {}
        pulse_queue = Queue()
        result["name"] = queuename
        pulse_queue.enqueue(start)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group = self.universe["above"]
        time = length
        switchinterval = (show.bpminterval/len(group))*1000*4
        i = 1
        while time > 1:
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer, f"Dimmer reset"))
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["dimmer"], 255, "Dimmer on"))
            pulse_queue.enqueue(temp)
            if time - switchinterval < 0:
                switchinterval = time
            time -= switchinterval
            i += 1
            if i > len(group):
                i = 1
            if time > 1:
                pulse_queue.enqueue(switchinterval)
        result["queue"] = pulse_queue
        return result
    
    def idle(self, name, show, length=30000.0, start=0, queuename="idle"):
        result = {}
        idle_queue = Queue()
        result["name"] = queuename
        idle_queue.enqueue(start)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group = self.universe["above"]
        group2 = self.universe["flood"]
        time = length
        temp = []
        for fixture in group2.values():
            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
        for fixture in group.values():
            temp.append(self._setfixture(fixture["id"], fixture["tilt"], 90, f"Tilt to 90"))
            temp.append(self._setfixture(fixture["id"], fixture["pan"], 127, f"Pan to 127"))
            temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 140, f"Dimmer on 200"))
        idle_queue.enqueue(temp)
        result["queue"] = idle_queue
        return result
    
    def pause(self, length, type="blackout", start=0, queuename="pause"):
        result = {}
        result["name"] = queuename
        if type == "blackout":
            blackout_queue = Queue()
            blackout_queue.enqueue(start)
            temp = []
            temp.append(self._blackout("on", f"Blackout for {length} seconds"))
            blackout_queue.enqueue(temp)
            wait = length * 1000
            blackout_queue.enqueue(wait)
            temp = []
            temp.append(self._blackout("off", f"Blackout off"))
            blackout_queue.enqueue(temp)
            result["queue"] = blackout_queue

        if type == "beams":
            result = {}
            result["name"] = "pause"
            group = self.universe["above"]
            floodgroup = self.universe["flood"]
            beam_queue = Queue()
            beam_queue.enqueue(0)
            temp = []
            wait = 200
            time = length*1000 - wait
            speed = self.calculate_tilt_speed(group["1"], 74, time/1000)
            for fixture in floodgroup.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
            beam_queue.enqueue(temp)
            beam_queue.enqueue(0)
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
                temp.append(self._setfixture(fixture["id"], fixture["movespeed"], 255, f"Set move speed"))
                temp.append(self._setfixture(fixture["id"], fixture["pan"], 127, f"Pan to 127"))
                temp.append(self._setfixture(fixture["id"], fixture["tilt"], 100, f"Tilt to 100"))
            beam_queue.enqueue(temp)
            beam_queue.enqueue(wait)
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["movespeed"], speed, f"Set move speed"))
                temp.append(self._setfixture(fixture["id"], fixture["tilt"], 0, f"Tilt to 0"))
            beam_queue.enqueue(temp)
            beam_queue.enqueue(time)
            result["queue"] = beam_queue
        return result

    def combine(self, queues, bpm):
        command_queues = {}
        for queue in queues:
            command_queues[queue["name"]] = queue["queue"]
        times = {}
        for queue in queues:
            times[queue["name"]] = queue["queue"].dequeue()
        index = 1
        while len(times) > 0:
            do_not_execute = []
            min_time = min(times.values())
            min_queues = [k for k, v in times.items() if v == min_time]
            if 'flood' in min_queues:
                q = ('flood', min_time)
            else:
                q = min_queues[0], min_time
            if "pause" not in q[0]:
                index2 = int(re.search(r'\d+$', q[0]).group())
                if index2 > index:
                    index = index2
                elif index2 < index:
                    for key in command_queues:
                        key_index = int(re.search(r'\d+$', key).group())
                        if key_index < index:
                            do_not_execute.append(key)
            queue = command_queues[q[0]]
            self._write(self._wait(q[1], f"Wait for {q[0]}"))
            for name in times:
                times[name] -= q[1]+(3.4)

                if times[name] < 0:
                    times[name] = 0
            commands = queue.dequeue()
            if commands == None:
                del times[q[0]]
                del command_queues[q[0]]
                continue
            if q[0] not in do_not_execute:
                for command in commands:
                    self._write(command)
            wait = queue.dequeue()
            if wait:
                times[q[0]] = wait

    def generate_segment(self, name, show, length, intensity, pauses):
        queues = []
        if intensity == 0:
            for pause in pauses:
                pausename = f"pause{str(pause[0])[:5]}"
                print(pausename)
                queues.append(self.pause((pause[1] - pause[0]), type="beams", start=pause[0]*1000, pausename=pausename))
            queues.append(self.idle(name, show=show))
            queues.append(self.pulse(name, show=show, length=length))
        elif intensity == 1:
            for pause in pauses:
                pausename = f"pause{str(pause[0])[:5]}"
                print(pausename)
                queues.append(self.pause((pause[1] - pause[0]), type="blackout", start=pause[0]*1000, pausename=pausename))
            queues.append(self.alternate(name, show=show, length=length))
            queues.append(self.spin(name, length=length))
            queues.append(self.flood(name, length=length))
        self.combine(queues)


    def generate_show(self, name):
        queues = []
        show = self.create_show(name)
        sections = show.struct["chorus_sections"]
        segments = show.struct["segments"]

        pauses = show.struct["silent_ranges"]
        for pause in pauses:
            pause_start = pause[0] / 43
            pause_end = pause[1] / 43
            pausename = f"pause{str(pause[0])[:5]}"
            queues.append(self.pause((pause_end - pause_start), type="blackout", queuename=pausename, start=pause_start*1000))

        i = 0 
        if segments[0]["label"] == "start":
            i += 1
        
        for i in range(i, len(segments)):
            found = False
            for section in sections:

                if segments[i]["start"] == section["seg_start"]:

                    print("found")
                    found = True
                    length = (segments[i]["end"] - segments[i]["start"])*1000

                    queues.append(self.alternate(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"alternate{i}"))
                    queues.append(self.swing(name, show, length=length, start=segments[i]["start"]*1000, queuename=f"spin{i}"))
                    queues.append(self.flood(name, length=length, start=segments[i]["start"]*1000, queuename=f"flood{i}"))

                    i += 1
                    break

            if found == False:
                
                length = (segments[i]["end"] - segments[i]["start"])*1000
                queues.append(self.idle(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"idle{i}"))
                queues.append(self.pulse(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"pulse{i}"))
                
                i += 1
        self.combine(queues, show.bpm)
                
            

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
        self.bpm = struct["bpm"]
        self.mp3_path = song_data["file"]
        self.bpminterval = 60 / (struct["bpm"]*1.02)
        self.beatinterval = 60 / struct["bpm"]
