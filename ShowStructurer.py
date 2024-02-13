import DataService
import QLCService
import re
import random
from QLCService import QXWHandler

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe = {"abovewash": 
                         {"1": {"id": 1, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                                "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 110},

                                   "2": {"id": 2, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                                         "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 110},

                                   "3": {"id": 3, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                                         "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 110},

                                   "4": {"id": 4, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                                          "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 110}
                                   },

                        "strobe": 
                        {"1": {"id": 0, "dimmer": 0, "colortype": "seperate", "colorchannels": {"red": 2, "green": 3, "blue": 4}, "strobe": 1, "stroberange": (130, 249),
                                "shutter": 1, "shutters": {"open": 0, "closed": 7}, "nicestrobe": 110},

                         "2": {"id": 5, "dimmer": 0, "colortype": "seperate", "colorchannels": {"red": 2, "green": 3, "blue": 4}, "strobe": 1, "stroberange": (130, 249),
                                "shutter": 1, "shutters": {"open": 0, "closed": 7}, "nicestrobe": 110}
                        }
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
        speed = 255 - (dmx_angle * time)
        if speed < 0:
            speed = 0
        return speed
    
    def calculate_tilt_speed(self, fixture, angle, time):
        range = fixture["tiltrange"]
        rate = 255 / range
        dmx_angle = angle * rate
        speed = 255 - (dmx_angle * time)
        if speed < 0:
            speed = 0
        return speed
    
    def legacy_calculate_tilt_speed(self, fixture, angle, time):
            range = fixture["tiltrange"]
            rate = 255 / range
            dmx_angle = angle * rate
            return dmx_angle / time
    
    def legacy_calculate_pan_speed(self, fixture, angle, time):
        range = fixture["panrange"]
        rate = 255 / range
        dmx_angle = angle * rate
        return dmx_angle / time
    
    def calculate_colors(self, fixture, color):
        temp = []
        if "colors" in fixture:
            color_1 = fixture["colors"][color]
            temp.append(self._setfixture(fixture["id"], fixture["color"], color_1, f"Dimmer on"))
        elif fixture["colortype"] == "seperate":
            if color == "white":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 255, f"Dimmer on"))
            elif color == "red":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 0, f"Dimmer on"))
            elif color == "green":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 0, f"Dimmer on"))
            elif color == "blue":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 255, f"Dimmer on"))
            elif color == "pink":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 255, f"Dimmer on"))
            elif color == "yellow":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 0, f"Dimmer on"))
            elif color == "cyan":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 255, f"Dimmer on"))
            elif color == "orange":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 255, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 100, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 0, f"Dimmer on"))
            elif color == "purple":
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], 100, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], 0, f"Dimmer on"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], 255, f"Dimmer on"))
        return temp
    

    def alternate(self, name, show, length=30000.0, start=0, queuename="alternate0"):
        result = {}
        light_queue = Queue()
        result["name"] = queuename
        light_queue.enqueue(start)
        group1 = self.universe["abovewash"]
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

    def spin(self, name, show, length=30000.0, start=0, queuename="spin0"):
        result = {}
        spin_queue = Queue()
        result["name"] = queuename
        spin_queue.enqueue(start)
        self.shows[name] = show
        group1 = self.universe["abovemoving"]
        time = length
        start_tilt = 23
        start_pan = 126
        fixture = group1["1"]
        pan_offset = 35
        tilt_offset = int(pan_offset * (fixture["tiltrange"]/fixture["panrange"])) + 10
        print(tilt_offset)
        circle_time = 2 * (60/show.bpm)
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
    
    def swing(self, name, show, length=30000.0, start=0, queuename="swing0"):
        result = {}
        swing_queue = Queue()
        result["name"] = queuename
        swing_queue.enqueue(start)
        group = self.universe["abovemoving"]
        time = length
        beatinterval = show.bpminterval
        print(f"beat{beatinterval}")
        angle = 70
        swingtime = beatinterval
        print(f"swingtime{swingtime}")
        mod1 = 0.15
        mod2 = 0.2
        mod3 = 0.3
        angles = [angle*mod1, angle*mod2, angle*mod3, angle*mod2, angle*mod1]
        tiltspeed1 = self.calculate_tilt_speed(group["1"], angles[1], swingtime*mod1)
        tiltspeed2 = self.calculate_tilt_speed(group["1"], angles[2], swingtime*mod2)
        tiltspeed3 = self.calculate_tilt_speed(group["1"], angles[3], swingtime*mod3)
        waits = [swingtime*mod1*1000, swingtime*mod2*1000, swingtime*mod3*1000, swingtime*mod2*1000, swingtime*mod1*1000]
        speeds = [tiltspeed1, tiltspeed2, tiltspeed3, tiltspeed2, tiltspeed1]
        switch = 0
        switch2 = 1
        for fixture in group.values():
            line = self._setfixture(fixture["id"], fixture["pan"], 127, f"Set pan")
            swing_queue.enqueue([line])
            swing_queue.enqueue(0)
        while time > 1:
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["movespeed"], speeds[switch], f"Set move speed"))

                if switch2 == 1 and switch == 0:
                    if int(fixture["id"]) % 2 == 0:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 0))
                    else:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], angle))
                elif switch2 == 0 and switch == 0:
                    if int(fixture["id"]) % 2 == 0:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], angle))
                    else:
                        temp.append(self._setfixture(fixture["id"], fixture["tilt"], 0))
            wait = waits[switch]
            switch += 1
            if switch > 4:
                switch = 0
                if switch2 == 1:
                    switch2 = 0
                else:
                    switch2 = 1
            swing_queue.enqueue(temp)
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                swing_queue.enqueue(wait)
        result["queue"] = swing_queue
        return result

    def flood(self, name, show, interval=None, length=30000.0, start=0, queuename="flood0", color="white"):
        result = {}
        flood_queue = Queue()
        result["name"] = queuename
        flood_queue.enqueue(start)
        struct, song_data = self.get_songdata(name)
        self.shows[name] = show
        floodexists = False
        groups = []
        if "flood" in self.universe:
            group1 = self.universe["flood"]
            floodexists = True
            groups.append(group)
        else:
            if "abovewash" in self.universe:
                group1 = self.universe["abovewash"]
                groups.append(group1)
            if "strobe" in self.universe:
                group2 = self.universe["strobe"]
                groups.append(group2)
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
            else:
                lightvalue -= 255/(1000*beatinterval/wait)
            for group in groups:
                for fixture in group.values():
                    colors = self.calculate_colors(fixture, color)
                    temp += colors
                    temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], lightvalue, f"{time}"))
            flood_queue.enqueue(temp)
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                flood_queue.enqueue(wait)
        result["queue"] = flood_queue
        return result
    
    def pulse(self, name, show, intervalmod=1, dimmer1=255, dimmer2=100, color1="white", color2="white", length=30000.0, start=0, queuename="pulse0"):
        result = {}
        pulse_queue = Queue()
        result["name"] = queuename
        pulse_queue.enqueue(start)
        self.shows[name] = show
        group = self.universe["abovewash"]
        time = length
        switchinterval = (show.bpminterval/len(group))*1000*4/intervalmod
        i = 1
        while time > 1:
            temp = []
            for fixture in group.values():
                color_commands = self.calculate_colors(fixture, color1)
                temp += color_commands
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer1, f"Dimmer reset"))
            color_commands = self.calculate_colors(group[str(i)], color2)
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["dimmer"], dimmer2, "Dimmer off"))
            temp += color_commands
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
    
    def idle(self, name, show, length=30000.0, start=0, queuename="idle0"):
        result = {}
        idle_queue = Queue()
        result["name"] = queuename
        idle_queue.enqueue(start)
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        group = None
        if "abovemoving" in self.universe:
            group = self.universe["abovemoving"]
        else:
            if "abovewash" in self.universe:
                group = self.universe["abovewash"]
        if "flood" in self.universe:
            group2 = self.universe["flood"]
        elif "strobe" in self.universe:
            group2 = self.universe["strobe"]
        time = length
        temp = []
        for fixture in group2.values():
            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 100, f"Dimmer off"))
        if group is not None:
            for fixture in group.values():
                if "abovemoving" in self.universe:
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
            group = self.universe["abovemoving"]
            othergroups = []
            if "abovewash" in self.universe:
                othergroups.append(self.universe["abovewash"])
            if "flood" in self.universe:
                othergroups.append(self.universe["flood"])
            if "strobe" in self.universe:
                othergroups.append(self.universe["strobe"])
            beam_queue = Queue()
            beam_queue.enqueue(0)
            temp = []
            wait = 200
            time = length*1000 - wait
            speed = self.calculate_tilt_speed(group["1"], 74, time/1000)
            for group in othergroups:
                for fixture in group.values():
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
            beam_queue.enqueue(temp)
            beam_queue.enqueue(0)
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
                temp.append(self._setfixture(fixture["id"], fixture["movespeed"], 0, f"Set move speed"))
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
    
    def reset_position(self, queuename="reset1"):
        result = {}
        result["name"] = queuename
        queue = Queue()
        queue.enqueue(0)
        group = self.universe["abovemoving"]
        temp = []
        pan = 127
        tilt = 65
        for fixture in group.values():
            temp.append(self._setfixture(fixture["id"], fixture["movespeed"], 0, f"Set move speed"))
            temp.append(self._setfixture(fixture["id"], fixture["pan"], pan, f"Pan to 127"))
            temp.append(self._setfixture(fixture["id"], fixture["tilt"], tilt, f"Tilt to 50"))
            temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"Dimmer on 200"))
        queue.enqueue(temp)
        wait = 150
        queue.enqueue(wait)
        result["queue"] = queue
        return result
    
    def randomstrobe(self, name, show, length=30000.0, start=0, queuename="strobe0", color="white"):
        result = {}
        strobe_queue = Queue()
        result["name"] = queuename
        strobe_queue.enqueue(start)
        groups = []
        if "abovewash" in self.universe:
            group1 = self.universe["abovewash"]
            groups.append(group1)
        if "strobe" in self.universe:
            group2 = self.universe["strobe"]
            groups.append(group2)
        time = length
        fixtures = []
        fixturedimmers = {}
        indexes = [0, 0]
        for group in groups:
            fixtures.append(group.values())
            for fixture in group.values():
                fixturedimmers[fixture["id"]] = 255
        while time > 1:
            for set in fixtures:
                print(set)
                number = random.randint(0, len(set)-1)
                while number == indexes[fixtures.index(set)]:
                    number = random.randint(0, len(set)-1)
                print(number)
                temp = []
                j = 0
                for fixture in set:
                    if j == number:
                        colors = self.calculate_colors(fixture, color)
                        temp += colors
                        temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"Dimmer on"))
                        temp.append(self._setfixture(fixture["id"], fixture["strobe"], fixture["nicestrobe"], f"Strobe on"))
                        fixturedimmers[fixture["id"]] = 255
                    elif fixturedimmers[fixture["id"]] > 0:
                        temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
                        fixturedimmers[fixture["id"]] = 0
                    j += 1
                indexes[fixtures.index(set)] = number
                wait = 50
                if time - wait < 0:
                    wait = time
                time -= wait
                if time > 1:
                    strobe_queue.enqueue(temp)
                    strobe_queue.enqueue(wait)
                else:
                    result["queue"] = strobe_queue
                    return result


    def combine(self, queues, qxwhandler, showname, script_name):
        segment = []
        command_queues = {}
        for queue in queues:
            command_queues[queue["name"]] = queue["queue"]
        times = {}
        for queue in queues:
            times[queue["name"]] = queue["queue"].dequeue()
        index = 0
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
            segment.append(self._wait(q[1], f"Wait for {q[0]}"))
            for name in times:
                times[name] -= (q[1] + 3)

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
                    segment.append(command)
            wait = queue.dequeue()
            if wait:
                times[q[0]] = wait
        qxwhandler.add_script(showname, segment, script_name)

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

    def generate_show(self, name, file_path="Newsetup.qxw"):
        qxw_handler = QXWHandler(file_path)
        qxw_handler.create_copy(name)
        queues = []
        show = self.create_show(name)
        # queues.append(self.randomstrobe(name, show, length=5000))
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
            queues = []
            found = False
            for section in sections:

                if segments[i]["start"] == section["seg_start"]:

                    print("found")
                    found = True
                    length = (segments[i]["end"] - segments[i]["start"])*1000

                    queues.append(self.alternate(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"alternate{i}"))
                    if "abovemoving" in self.universe:
                        if segments[i]["label"] == show.struct["focus"]["first"]:
                            queues.append(self.spin(name, show, length=length, start=segments[i]["start"]*1000, queuename=f"spin{i}"))
                        else:
                            queues.append(self.swing(name, show, length=length, start=segments[i]["start"]*1000, queuename=f"swing{i}"))
                    queues.append(self.flood(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"flood{i}"))

                    break

            if found == False:
                
                length = (segments[i]["end"] - segments[i]["start"])*1000
                queues.append(self.idle(name, show=show, length=length, start=segments[i]["start"]*1000, queuename=f"idle{i}"))
                queues.append(self.pulse(name, show=show, length=length, start=segments[i]["start"]*1000, color1="green", color2="red", queuename=f"pulse{i}"))
            
            self.combine(queues, qxw_handler, name, segments[i]["start"])
            i += 1
                
            

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
        self.bpminterval = 60 / (struct["bpm"])
        self.beatinterval = 60 / struct["bpm"]
