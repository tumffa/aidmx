import re
import random
import math
from scipy import interpolate

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe = {}
        self.universe["abovewash"] = {
            "1": {"id": 1, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250},
            "2": {"id": 2, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250},
            "3": {"id": 3, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250},
            "4": {"id": 4, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": (20, 255),
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250}
        }

        self.universe["strobe"] = {
            "1": {"id": 0, "dimmer": 0, "colortype": "seperate", "colorchannels": {"red": 2, "green": 3, "blue": 4}, "strobe": 1, "stroberange": (130, 249),
                  "shutter": 1, "shutters": {"open": 0, "closed": 7}, "nicestrobe": 211},
            "2": {"id": 5, "dimmer": 0, "colortype": "seperate", "colorchannels": {"red": 2, "green": 3, "blue": 4}, "strobe": 1, "stroberange": (130, 249),
                  "shutter": 1, "shutters": {"open": 0, "closed": 7}, "nicestrobe": 211}
        }

        # Dimmer map for help with seperating dimmer commands
        self.fixture_dimmer_map = {}
        for group_name, group in self.universe.items():
            for fixture_key, fixture in group.items():
                fixture_id = fixture["id"]
                dimmer_channel = fixture["dimmer"]
                self.fixture_dimmer_map[fixture_id] = dimmer_channel

        self.dimmer_update_fq = 15 # ms
        self.wait_adjustment = 0.06  # Adjust wait time to help with lag
        self.pause_wait_adjustment = 0.02  # Adjust wait time to help with lag
        self.command_counter_multi = 4 # Pattern wait times are adjusted by dimmer commands x multi

    def adjusted_wait(self, time, is_pause=False):
        if is_pause:
            return time - max(1, int(math.floor(time * self.pause_wait_adjustment)))
        return time - max(1, int(math.floor(time * self.wait_adjustment)))

    def get_songdata(self, name):
        struct = self.dm.get_struct_data(name)
        song_data = self.dm.get_song(name)
        return struct, song_data
    
    def create_show(self, name):
        struct, song_data = self.get_songdata(name)
        show = Show(name, struct, song_data)
        self.shows[name] = show
        return show
    
    def _light_strength_envelope_function(self, light_strength_envelope):
        """
        Converts a discrete light strength envelope into a callable function.
        This allows calculating envelope values at any time without storing large arrays.
        
        Args:
            light_strength_envelope: Dictionary with 'times' and 'values' lists
            
        Returns:
            A function that takes a time value and returns the envelope strength at that time
        """
        if not light_strength_envelope:
            return lambda t: 1.0
            
        env_times = light_strength_envelope.get("times", [])
        env_values = light_strength_envelope.get("values", [])
        segment_start = light_strength_envelope.get("segment_start", 0)
        segment_end = light_strength_envelope.get("segment_end", 0)
        
        if not env_times or not env_values:
            return lambda t: 1.0
        
        # Create interpolation function (linear interpolation)
        interp_func = interpolate.interp1d(
            env_times, 
            env_values,
            bounds_error=False,     # Don't raise error for out-of-bounds
            fill_value=(0.5, 0.5)   # Use 0.5 (baseline) for points outside range
        )
        
        # Return a closure function that handles the interpolation
        def envelope_function(t):
            # Handle out-of-segment values
            if t < segment_start or t > segment_end:
                return 0.5  # Baseline strength
                
            # Use interpolation for values within range
            return float(interp_func(t))
            
        return envelope_function

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
    
    def _execute(self, command, comment=""):
        return f"systemcommand:{command} //{comment}"

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
            temp.append(self._setfixture(fixture["id"], fixture["color"], color_1, f"color"))
        elif fixture["colortype"] == "seperate":
            color_map = {
                "white": {"red": 255, "green": 255, "blue": 255},
                "red": {"red": 255, "green": 0, "blue": 0},
                "green": {"red": 0, "green": 255, "blue": 0},
                "blue": {"red": 0, "green": 0, "blue": 255},
                "pink": {"red": 255, "green": 0, "blue": 255},
                "yellow": {"red": 255, "green": 255, "blue": 0},
                "cyan": {"red": 0, "green": 255, "blue": 255},
                "orange": {"red": 255, "green": 100, "blue": 0},
                "purple": {"red": 100, "green": 0, "blue": 255}
            }

            if color in color_map:
                channels = color_map[color]
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], channels["red"], f"color"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], channels["green"], f"color"))
                temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], channels["blue"], f"color"))
        return temp
    
    def slow_flash(self, name, show, length=30000.0, start=0, queuename="slowflash0"):
        # an effect that every 4 beats flashes a new colour that fades off
        result = {}
        slowflash_queue = Queue()
        result["name"] = queuename
        slowflash_queue.enqueue(start)
        beatinterval = show.beatinterval
        time = length
        group = {}
        if "abovewash" in self.universe:
            group.update(self.universe["abovewash"])
        if "strobe" in self.universe:
            for key, value in self.universe["strobe"].items():
                if key not in group:
                    group[key] = value
        
        colors = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        iterations = 0
        color = random.choice(colors)
        
        # Calculate total steps needed for one full fade cycle (0 to 255 to 0)
        beat_duration_ms = beatinterval * 1000
        steps_per_beat = beat_duration_ms / 15  # With 15ms fixed timing
        total_steps = int(16 * steps_per_beat)  # 16 beats for full cycle
        
        # Fixed wait time
        wait = self.dimmer_update_fq
        
        # Calculate how much to increment dimmer per step
        step_size = 50.0 / (total_steps / 4)  # Divide by 4 to complete cycle in 1/4 of total time
        
        while time > 1:
            if iterations >= 50:
                iterations = 0
                color = random.choice(colors)
            
            temp = []
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], "shutters"))
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255*(iterations/50), f"dimmer"))
                temp += self.calculate_colors(fixture, color)
            
            iterations += step_size  # Increment by step size for smoother transition
            slowflash_queue.enqueue(temp)
            
            if time - wait < 0:
                wait = time
            time -= wait
            
            if time > 1:
                slowflash_queue.enqueue(wait)
        
        result["queue"] = slowflash_queue
        return result

    def alternate(self, name, show, length=30000.0, start=0, queuename="alternate0"):
        result = {}
        light_queue = Queue()
        result["name"] = queuename
        light_queue.enqueue(start)
        if "abovewash" in self.universe:
            group1 = self.universe["abovewash"]
        else:
            group1 = self.universe["strobe"]
        beatinterval = show.beatinterval
        time = length
        switch = 0
        colors = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        color1 = random.choice(colors)
        colors = [color for color in colors if color != color1]
        color2 = random.choice(colors)
        fixtures_amount = len(group1)
        while time > 1:
            temp = []
            for fixture in group1.values():
                if switch == 0:
                    if int(fixture["id"]) > fixtures_amount/2:
                        temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], "Open shutters"))
                        temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}"))
                        temp += self.calculate_colors(fixture, color1)
                    else:
                        line = self._setfixture(fixture["id"], fixture["dimmer"], 0, f"{time}")
                        temp.append(line)
                else:
                    if int(fixture["id"]) <= fixtures_amount/2:
                        temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], "Open shutters"))
                        temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"{time}"))
                        temp += self.calculate_colors(fixture, color2)
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
        beatinterval = show.beatinterval
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
            # if "abovewash" in self.universe:
            #     group1 = self.universe["abovewash"]
            #     groups.append(group1)
            if "strobe" in self.universe:
                group2 = self.universe["strobe"]
                groups.append(group2)
        time = length
        if interval:
            beatinterval = interval
        else:
            beatinterval = show.beatinterval
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
    
    def alternate_flood(self, name, show, length=30000.0, start=0, queuename="alternateflood0"):
        result = {}
        alternateflood_queue = Queue()
        result["name"] = queuename
        alternateflood_queue.enqueue(start)
        struct, song_data = self.get_songdata(name)
        self.shows[name] = show
        if "flood" in self.universe:
            group = self.universe["flood"]
        elif "strobe" in self.universe:
            group = self.universe["strobe"]
        divider = len(group) // 2
        time = length
        beatinterval = show.beatinterval
        wait = beatinterval * 1000
        j = 0
        which = 0
        colors = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        last_color = None
        while time > 1:
            if j == 4:
                j = 0
            if j == 0:
                color1 = random.choice(colors)
                while color1 == last_color:
                    color1 = random.choice(colors)
                last_color = color1
            temp = []
            for i in range(len(group)):
                if int(list(group.keys())[i]) <= divider:
                    if which == 0:
                        colorcommands = self.calculate_colors(group[str(i+1)], color1)
                        temp += colorcommands
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["shutter"], group[str(i+1)]["shutters"]["open"], f"Open shutters"))
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["dimmer"], 80, f"{time}"))
                    else:
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["dimmer"], 0, f"{time}"))
                else:
                    if which == 1:
                        colorcommands = self.calculate_colors(group[str(i+1)], color1)
                        temp += colorcommands
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["shutter"], group[str(i+1)]["shutters"]["open"], f"Open shutters"))
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["dimmer"], 80, f"{time}"))
                    else:
                        temp.append(self._setfixture(group[str(i+1)]["id"], group[str(i+1)]["dimmer"], 0, f"{time}"))
            which = 1 - which
            j += 1
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                alternateflood_queue.enqueue(temp)
                alternateflood_queue.enqueue(wait)
        result["queue"] = alternateflood_queue
        return result

    def pulse(self, name, show, intervalmod=1, dimmer1=255, dimmer2=50, color1="white", color2="white", length=30000.0, start=0, queuename="pulse0"):
        result = {}
        pulse_queue = Queue()
        result["name"] = queuename
        pulse_queue.enqueue(start)
        self.shows[name] = show
        group = self.universe["abovewash"]
        time = length
        switchinterval = (show.beatinterval/len(group))*1000*4/intervalmod
        i = 1
        while time > 1:
            temp = []
            for fixture in group.values():
                color_commands = self.calculate_colors(fixture, color1)
                temp += color_commands
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer2, f"Dimmer reset"))
            color_commands = self.calculate_colors(group[str(i)], color2)
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["dimmer"], dimmer1, "Dimmer off"))
            temp += color_commands
            if time - switchinterval < 0:
                switchinterval = time
            time -= switchinterval
            i += 1
            if i > len(group):
                i = 1
            if time > 1:
                pulse_queue.enqueue(temp)
                pulse_queue.enqueue(switchinterval)
        result["queue"] = pulse_queue
        return result
    
    def fastpulse(self, name, show, intervalmod=4, dimmer1=255, dimmer2=0, color1="all", color2=None, length=30000.0, start=0, queuename="fastpulse0"):
        result = {}
        fastpulse_queue = Queue()
        result["name"] = queuename
        fastpulse_queue.enqueue(start)
        if "abovewash" in self.universe:
            group = self.universe["abovewash"]
        elif "strobe" in self.universe:
            group = self.universe["strobe"]
        time = length
        switchinterval = (show.beatinterval/len(group))*1000*4/intervalmod
        i = 1
        if color1 == "all":
            colors = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        else:
            colors = [color1]
        last_color = None
        while time > 1:
            temp = []
            if i == 1:
                color1 = random.choice(colors)
                while color1 == last_color:
                    color1 = random.choice(colors)
                last_color = color1
            for fixture in group.values():
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer2, f"Dimmer reset"))
            color_commands = self.calculate_colors(group[str(i)], color1)
            temp += color_commands
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["shutter"], group[str(i)]["shutters"]["open"], f"Open shutters"))
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["dimmer"], dimmer1, "Dimmer off"))
            if time - switchinterval < 0:
                switchinterval = time
            time -= switchinterval
            i += 1
            if i > len(group):
                i = 1
            if time > 1:
                fastpulse_queue.enqueue(temp)
                fastpulse_queue.enqueue(switchinterval)
        result["queue"] = fastpulse_queue
        return result
    
    def side_to_side(self, name, show, intervalmod=4, dimmer1=255, dimmer2=25, color1="random", color2=None, length=30000.0, start=0, queuename="sidetoside0"):
        result = {}
        sidetoside_queue = Queue()
        result["name"] = queuename
        sidetoside_queue.enqueue(start)
        groups = []
        time = length
        if "abovewash" in self.universe:
            group1 = self.universe["abovewash"]
            groups.append(group1)
        if "strobe" in self.universe:
            group2 = self.universe["strobe"]
            groups.append(group2)
        fixtures = []
        for group in groups:
            for fixture in group.values():
                fixtures.append(fixture)
        fixtures = sorted(fixtures, key=lambda item: item['id'])
        colorspectrum = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        if color1 != "random":
            color1 = color1
            color2 = color2
        else:
            color2 = random.choice(colorspectrum)
            color1 = color2
        switchinterval = (show.beatinterval/(len(groups)+0.55))*1000*2/intervalmod
        switch = 0
        changed = []
        i = 0
        while time > 1:
            if switch == 0 and i == 0:
                color1 = random.choice(colorspectrum)
                while color1 == color2:
                    color1 = random.choice(colorspectrum)
            elif switch == 1 and i == len(fixtures)-1:
                color2 = random.choice(colorspectrum)
                while color1 == color2:
                    color2 = random.choice(colorspectrum)
            temp = []
            for fixture in fixtures:
                if switch == 0:
                    if fixture["id"] <= i:
                        color_commands = self.calculate_colors(fixture, color1)
                        temp += color_commands
                elif switch == 1:
                    if fixture["id"] >= i:
                        color_commands = self.calculate_colors(fixture, color2)
                        temp += color_commands
                if fixture["id"] == i:
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer1, f"Dimmer off"))
                    temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                else:
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer2, f"Dimmer reset"))
                    temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                    
            if switch == 0:
                i += 1
            else:
                i -= 1
            if i == len(fixtures) - 1:
                switch = 1
            elif i == 0:
                switch = 0
            if time - switchinterval < 0:
                switchinterval = time
            time -= switchinterval
            if time > 1:
                sidetoside_queue.enqueue(temp)
                sidetoside_queue.enqueue(switchinterval)
        result["queue"] = sidetoside_queue
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
        temp = []
        # for fixture in group2.values():
        #     temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 100, f"Dimmer off"))
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
    
    def randomstrobe(self, name, show, length=30000.0, start=0, queuename="strobe0", strobecolor="white", waittime=50):
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
        colorspectrum = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        if strobecolor != "random":
            color = strobecolor
        last_color = None
        for group in groups:
            fixtures.append(group.values())
            for fixture in group.values():
                fixturedimmers[fixture["id"]] = 255
        while time > 1:
            for set in fixtures:
                if strobecolor == "random":
                    color = random.choice(colorspectrum)
                    while color == last_color:
                        color = random.choice(colorspectrum)
                    last_color = color
                number = random.randint(0, len(set)-1)
                while number == indexes[fixtures.index(set)]:
                    number = random.randint(0, len(set)-1)
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
                wait = waittime
                if time - wait < 0:
                    wait = time
                time -= wait
                if time > 1:
                    strobe_queue.enqueue(temp)
                    strobe_queue.enqueue(wait)
        result["queue"] = strobe_queue
        return result
                
    def blind(self, name, show, length=10000, start=0, queuename="blind0", waittime = 50):
        result = {}
        blind_queue = Queue()
        result["name"] = queuename
        blind_queue.enqueue(start)
        groups = []
        if "abovewash" in self.universe:
            group1 = self.universe["abovewash"]
            groups.append(group1)
        if "strobe" in self.universe:
            group2 = self.universe["strobe"]
            groups.append(group2)
        if "flood" in self.universe:
            group3 = self.universe["flood"]
            groups.append(group3)
        if "blinders" in self.universe:
            group4 = self.universe["blinders"]
            groups.append(group4)
        time = length
        fixtures = []
        for group in groups:
            for fixture in group.values():
                fixtures.append(fixture)
        while time > 1:
            temp = []
            for fixture in fixtures:
                colorcommands = self.calculate_colors(fixture, "white")
                temp += colorcommands
                temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"Dimmer on"))
            wait = waittime
            if time - wait < 0:
                wait = time
            time -= wait
            if time > 1:
                blind_queue.enqueue(temp)
                blind_queue.enqueue(wait)
            else:
                result["queue"] = blind_queue
                return result

    def combine(self, queues, end_time=None, seperate_dimmer=True, light_strength_envelope=None, strobe_ranges=None):
        """
        Combines multiple command queues into a single sequence,
        keeping track of fixture dimmer values separately and applying envelope scaling.
        
        Args:
            queues: List of queue dictionaries
            start_time: Start time offset in seconds
            seperate_dimmer: If True, separate dimmer commands from main commands
            light_strength_envelope: Envelope for scaling dimmer values
            
        Returns:
            Tuple of (main_segment, dimmer_segment)
        """
        if light_strength_envelope:
            envelope_function =  envelope_function = self._light_strength_envelope_function(light_strength_envelope)
        else:
            envelope_function = lambda t: 1.0  # No scaling if no envelope is provided
        # Main pattern commands (excluding dimmer controls)
        segment = []

        # Separate list for dimmer commands
        segment_dimmers = []
        
        # Track the last set dimmer value for each fixture
        fixture_dimmers = {}  # {fixture_id: {"channel": channel, "original_value": value}}
        
        # Initialize time tracking
        if end_time is None:
            end_time = float('inf')
        current_time_ms = 0
        current_time_sec = current_time_ms / 1000.0
        
        # Update frequency for dimmer scaling during long waits (in ms)
        update_frequency_ms = self.dimmer_update_fq
        
        is_slowflash = False
        is_pause = False
        command_queues = {}
        for queue in queues:
            command_queues[queue["name"]] = queue["queue"]
            if "slowflash" in queue["name"]:
                is_slowflash = True
            if "pause" in queue["name"]:
                is_pause = True
        
        times = {}
        for queue in queues:
            times[queue["name"]] = queue["queue"].dequeue()
        
        index = 0
        pattern_started = False
        
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
            wait_time = q[1]

            # Check if current time in sec is in a strobe range
            # during this time commands are not executed
            # after strobe values are set back to saved state
            in_strobe_range = False
            if strobe_ranges and not is_pause:
                for strobe_range in strobe_ranges:
                    if strobe_range[0] <= current_time_sec <= strobe_range[1]:
                        in_strobe_range = True
                        current_strobe_range_end_sec = strobe_range[1]
                        break
            
            # Check if we need to insert dynamic dimmer updates during this wait
            dimmer_condition = (
                wait_time > update_frequency_ms and
                light_strength_envelope and
                fixture_dimmers and
                seperate_dimmer and
                pattern_started and 
                not in_strobe_range and 
                not is_pause
            )

            found_updates = False

            if dimmer_condition:
                active_ranges = light_strength_envelope.get("active_ranges", [])
                # Add dynamic dimmer updates during this wait period
                found_updates= self._add_dynamic_dimmer_updates(
                    segment, segment_dimmers, fixture_dimmers, 
                    wait_time, current_time_ms, update_frequency_ms, 
                    envelope_function=envelope_function,
                    active_ranges=active_ranges
                )

            segment.append(self._wait(wait_time, f"Wait for {q[0]} Pattern started: {pattern_started} is pause: {is_pause}"))
            if seperate_dimmer and not found_updates:
                segment_dimmers.append(self._wait(wait_time, f"First Wait for {q[0]} (dimmer), pattern started: {pattern_started}"))
            pattern_started = True
            
            current_time_ms += wait_time
            current_time_sec = current_time_ms / 1000.0
            
            # Process time updates
            for name in times:
                times[name] -= wait_time
                if times[name] < 0:
                    times[name] = 0
                    
            # Process commands
            commands = queue.dequeue()
            if commands == None:
                del times[q[0]]
                del command_queues[q[0]]
                continue
                    
            if q[0] not in do_not_execute:
                for command in commands:
                    match = re.search(r'setfixture:(\d+) ch:(\d+) val:(\d+)', command)
                    if match:
                        fixture_id, channel, value = match.groups()
                        fixture_id = int(fixture_id)
                        channel = int(channel)
                        original_value = int(value)
                        
                        if fixture_id not in fixture_dimmers:
                            fixture_dimmers[fixture_id] = {}
                        fixture_dimmers[fixture_id][channel] = original_value
                        
                        # Fast lookup using our map instead of searching nested dictionaries
                        is_dimmer_command = (fixture_id in self.fixture_dimmer_map and 
                                        channel == self.fixture_dimmer_map[fixture_id])
                        
                        if is_dimmer_command and seperate_dimmer:
                            # Get current envelope strength
                            strength = envelope_function(current_time_sec)
                            
                            if is_slowflash:
                                scaled_value = int(original_value * strength)
                            else:
                                scaled_value = original_value
                            
                            # Create new command with scaled value
                            dimmer_command = self._setfixture(fixture_id, channel, scaled_value,
                                                                f"Scaled value {scaled_value} from {original_value}")
                                
                            # Add to dimmer segment
                            if not in_strobe_range:
                                segment_dimmers.append(dimmer_command)
                        else:
                            if not in_strobe_range:
                                segment.append(command)
                    else:
                        if not in_strobe_range:
                            segment.append(command)

            if in_strobe_range:
                wait_time = (current_strobe_range_end_sec - current_time_sec) * 1000
                if wait_time > 0:
                    segment.append(self._wait(wait_time, f"Wait for strobe range"))
                    if seperate_dimmer:
                        segment_dimmers.append(self._wait(wait_time, f"Wait for strobe range (dimmer)"))
                for fixture_id in fixture_dimmers:
                    for channel, value in fixture_dimmers[fixture_id].items():
                        # Check if dimmer
                        if channel == self.fixture_dimmer_map.get(fixture_id, None):
                            # Get current envelope strength
                            strength = envelope_function(current_time_sec)
                            scaled_value = int(value * strength)
                            
                            dimmer_command = self._setfixture(fixture_id, channel, scaled_value,
                                                              f"Scaled value {scaled_value} from {value}")
                                
                            # Add to dimmer segment
                            segment_dimmers.append(dimmer_command)
                        else:
                            command = self._setfixture(fixture_id, channel, value, 
                                                       f"Set fixture {fixture_id} channel {channel} to {value}")
                            segment.append(command)

            # Get next wait period
            wait = queue.dequeue()
            if wait:
                times[q[0]] = wait
        
        return segment, segment_dimmers

    def _add_dynamic_dimmer_updates(self, segment, segment_dimmers, fixture_dimmers, 
                                wait_time, current_time_ms, update_frequency_ms,
                                envelope_function, active_ranges=None):
        """
        Adds dynamic dimmer updates during long wait periods based on envelope values.
        Uses pre-calculated active ranges for efficient updates.
        
        Args:
            segment: Main command segment
            segment_dimmers: Dimmer command segment
            fixture_dimmers: Dictionary of current fixture dimmer values
            wait_time: Total wait time in ms
            current_time_ms: Current time position in ms 
            update_frequency_ms: Update frequency in ms
            envelope_function: Function that returns envelope strength at given time
            active_ranges: List of active envelope ranges with start_ms and end_ms
        """
    
        # Use active ranges for precise updates
        wait_end_ms = current_time_ms + wait_time
        
        # Find active ranges that overlap with our current wait period
        relevant_ranges = [
            r for r in active_ranges 
            if r['end_ms'] >= current_time_ms and r['start_ms'] <= wait_end_ms
        ]
        
        if not relevant_ranges:
            return False
        
        # Sort ranges by start time
        relevant_ranges.sort(key=lambda r: r['start_ms'])
        
        # Process each range
        remaining_wait = wait_time
        update_time_ms = current_time_ms
        
        for range_idx, active_range in enumerate(relevant_ranges):
            # Calculate time to the start of this range if it's in the future
            if active_range['start_ms'] > update_time_ms:
                wait_to_range = active_range['start_ms'] - update_time_ms
                if wait_to_range > 0:
                    # Wait until the start of this active range
                    segment_dimmers.append(self._wait(self.adjusted_wait(wait_to_range), f"Wait until range at {active_range['start_ms']/1000}s"))
                    update_time_ms += wait_to_range
                    remaining_wait -= wait_to_range
                    
                    if remaining_wait <= 0:
                        break
            
            # Calculate how much time we spend in this active range
            range_end_capped = min(active_range['end_ms'], wait_end_ms)
            time_in_range = range_end_capped - update_time_ms
            
            # Apply updates while in the active range
            range_time_remaining = time_in_range
            while range_time_remaining > 0:
                update_chunk = min(update_frequency_ms, range_time_remaining)
                segment_dimmers.append(self._wait(self.adjusted_wait(update_chunk), f"Update in active range {active_range['max_value']:.2f}"))
                update_time_ms += update_chunk
                
                strength = envelope_function(update_time_ms / 1000.0)
                
                for fixture_id, info in fixture_dimmers.items():
                    for channel, value in info.items():
                        if channel == self.fixture_dimmer_map.get(fixture_id, None):
                            scaled_value = int(value * strength)
                            dimmer_command = self._setfixture(fixture_id, channel, scaled_value, 
                                                              f"Scaled value {scaled_value} from {value}")
                            segment_dimmers.append(dimmer_command)
                
                range_time_remaining -= update_chunk
                remaining_wait -= update_chunk
                
                if remaining_wait <= 0:
                    break
            
            # If we've used all our wait time, exit
            if remaining_wait <= 0:
                break
            
            # If there's another range coming up, wait between ranges
            if range_idx < len(relevant_ranges) - 1:
                next_range = relevant_ranges[range_idx + 1]
                wait_between_ranges = next_range['start_ms'] - update_time_ms
                wait_between_ranges = min(wait_between_ranges, remaining_wait)
                
                if wait_between_ranges > 0:
                    segment_dimmers.append(self._wait(self.adjusted_wait(wait_between_ranges), f"Wait between ranges"))
                    update_time_ms += wait_between_ranges
                    remaining_wait -= wait_between_ranges
        
        # If we have remaining wait time after all ranges, add final wait
        if remaining_wait > 0:
            segment_dimmers.append(self._wait(remaining_wait, f"Finished, wait for {remaining_wait}ms"))

        return True

    def generate_show(self, name, qxw, strobes=True):
        # delay for powershell command
        delay = 310 # milliseconds
        qxw.create_copy(name)
        scripts = []
        function_names = []
        show = self.create_show(name)
        sections = show.struct["chorus_sections"]
        segments = show.struct["segments"]
        
        # Add premade chasers
        self.add_chasers(name, show, qxw)

        onset_parts = None
        if strobes:
            onset_parts = show.struct["onset_parts"]
            for part in onset_parts:
                queues = []
                queues.append(self.randomstrobe(name, show, length=part[1]*1000-part[0]*1000, start=part[0]*1000 + delay, queuename=f"strobe{part[0]}", waittime=20))
                scripts.append(self.combine(queues)[0])
                function_names.append(f"strobe{part[0]}")

        queues = []
        pauses = show.struct["silent_ranges"]

        for pause in pauses:
            pause_start = pause[0] / 43
            pause_end = pause[1] / 43
            pausename = f"pause{str(pause[0])[:5]}"
            queues.append(self.pause((pause_end - pause_start), type="blackout", queuename=pausename, start=pause_start*1000 + delay))
        scripts.append(self.combine(queues)[0])
        function_names.append("pauses")

        i = 0
        if segments[0]["label"] == "start":
            i += 1
        onefocus = False
        if len(show.struct["focus"]) == 1:
            onefocus = True
        lastchaser = random.choice(["FastPulse", "SideToSide"])
        lastidle = random.choice(["Pulse", "SlowFlash"])

        for i in range(i, len(segments)):
            start_time = segments[i]["start"]*1000 + delay
            queues = []
            found = False
            subsegment = segments[i].get("subsegment", False)

            for section in sections:
                if segments[i]["start"] == section["seg_start"]:
                    found = True
                    length = (segments[i]["end"] - segments[i]["start"])*1000
                    types = ["alternate", "side_to_side"]
                    last_choice = None
                    if segments[i]["label"] == show.struct["focus"]["first"]:
                        if onefocus == False:
                            queues.append(self.fastpulse(name, show=show, length=length, start=start_time, queuename=f"fastpulse{i}"))
                        elif lastchaser == "FastPulse" and "abovewash" in self.universe:
                            if subsegment:
                                queues.append(self.fastpulse(name, show=show, length=length, start=start_time, queuename=f"fastpulse{i}"))
                                if "abovewash" in self.universe:
                                    queues.append(self.alternate_flood(name, show=show, length=length, start=start_time, queuename=f"alternateflood{i}"))
                                lastchaser = "FastPulse"
                            else:
                                queues.append(self.side_to_side(name, show=show, length=length, start=start_time, queuename=f"sidetoside{i}"))
                                lastchaser = "SideToSide"
                        elif lastchaser == "SideToSide" or "abovewash" not in self.universe:
                            if subsegment:
                                queues.append(self.side_to_side(name, show=show, length=length, start=start_time, queuename=f"sidetoside{i}"))
                                lastchaser = "SideToSide"
                            else:
                                queues.append(self.fastpulse(name, show=show, length=length, start=start_time, queuename=f"fastpulse{i}"))
                                if "abovewash" in self.universe:
                                    queues.append(self.alternate_flood(name, show=show, length=length, start=start_time, queuename=f"alternateflood{i}"))
                                lastchaser = "FastPulse"
                        print(f"Print added energetic {lastchaser} chaser ({segments[i]['start']}s - {segments[i]['end']}s) for {segments[i]['label']}")
                    else:
                        type = random.choice(types)
                        while type == last_choice:
                            type = random.choice(types)
                        last_choice = type
                        if "abovewash" not in self.universe:
                            type = "alternate"
                        if type == "alternate":
                            queues.append(self.alternate(name, show=show, length=length, start=start_time, queuename=f"alternateflood{i}"))
                        else:
                            queues.append(self.side_to_side(name, show=show, length=length, start=start_time, queuename=f"sidetoside{i}"))
                        print(f"Print added normal {last_choice} chaser ({segments[i]['start']}s - {segments[i]['end']}s) for {segments[i]['label']}")
                    break

            if found == False:
                length = (segments[i]["end"] - segments[i]["start"])*1000
                # queues.append(self.idle(name, show=show, length=length, start=start_time, queuename=f"idle{i}"))
                if segments[i-1]["label"] == segments[i]["label"]:
                    if lastidle == "Pulse" and "abovewash" in self.universe:
                        queues.append(self.pulse(name, show=show, dimmer1=100, dimmer2=30, length=length, start=start_time, color1="green", color2="red", queuename=f"pulse{i}"))
                        lastidle = "Pulse"
                    else:
                        queues.append(self.slow_flash(name, show=show, length=length, start=start_time, queuename=f"slowflash{i}"))
                        lastidle = "SlowFlash"
                else:
                    if lastidle == "Pulse" or "abovewash" not in self.universe:
                        queues.append(self.slow_flash(name, show=show, length=length, start=start_time, queuename=f"slowflash{i}"))
                        lastidle = "SlowFlash"
                    elif "abovewash" in self.universe:
                        queues.append(self.pulse(name, show=show, dimmer1=100, dimmer2=30, length=length, start=start_time, color1="green", color2="red", queuename=f"pulse{i}"))
                        lastidle = "Pulse"
                print(f"Print added idle {lastidle} chaser ({segments[i]['start']}s - {segments[i]['end']}s) for {segments[i]['label']}")

            end_time = segments[i]["end"]*1000 + delay
            light_strength_envelope = segments[i]["drum_analysis"]["light_strength_envelope"]
            segment_queue, segment_dimmers = self.combine(
                queues,
                end_time=end_time,
                light_strength_envelope=light_strength_envelope,
                strobe_ranges=onset_parts)
            scripts.append(segment_queue)
            scripts.append(segment_dimmers)
            function_names.append(str(segments[i]["start"]))
            function_names.append(str(segments[i]["start"]) + "_dimmers")
            i += 1
        qxw.add_track(scripts, name, function_names)

    def add_chasers(self, name, show, handler):
        handler.add_button(name, "BLACKOUT", "blackout", 1)

        queues = []
        queues.append(self.randomstrobe(name, show, length=2000))
        strobescript, _ = self.combine(queues, seperate_dimmer=False)
        strobeid = handler.add_script(name, strobescript, "FullStrobe")
        chaserid = handler.add_chaser(name, strobeid, "FullWhiteStrobe", duration=1900)
        handler.add_button(name, "FullWhiteStrobe", chaserid, 3)

        queues = []
        queues.append(self.blind(name, show, length=2000))
        blindscript, _  = self.combine(queues, seperate_dimmer=False)
        blindid = handler.add_script(name, blindscript, "blind")
        chaserid = handler.add_chaser(name, blindid, "Blind", duration=1900)
        handler.add_button(name, "BLIND", chaserid, 2)

        queues = []
        queues.append(self.randomstrobe(name, show, length=2000, strobecolor="random"))
        strobescript, _ = self.combine(queues, seperate_dimmer=False)
        strobeid = handler.add_script(name, strobescript, "RandomStrobe")
        chaserid = handler.add_chaser(name, strobeid, "RandomStrobe", duration=1900)
        handler.add_button(name, "RandomStrobe", chaserid, "R")

        queues = []
        queues.append(self.randomstrobe(name, show, length=2000, strobecolor="red"))
        strobescript, _ = self.combine(queues, seperate_dimmer=False)
        strobeid = handler.add_script(name, strobescript, "RedStrobe")
        chaserid = handler.add_chaser(name, strobeid, "RedStrobe", duration=1900)
        handler.add_button(name, "RedStrobe", chaserid, "T")

        queues = []
        queues.append(self.randomstrobe(name, show, length=2000, strobecolor="green"))
        strobescript, _ = self.combine(queues, seperate_dimmer=False)
        strobeid = handler.add_script(name, strobescript, "GreenStrobe")
        chaserid = handler.add_chaser(name, strobeid, "GreenStrobe", duration=1900)
        handler.add_button(name, "GreenStrobe", chaserid, "Y")

        queues = []
        queues.append(self.randomstrobe(name, show, length=2000, strobecolor="blue"))
        strobescript, _ = self.combine(queues, seperate_dimmer=False)
        strobeid = handler.add_script(name, strobescript, "BlueStrobe")
        chaserid = handler.add_chaser(name, strobeid, "BlueStrobe", duration=1900)
        handler.add_button(name, "BlueStrobe", chaserid, "U")

        queues = []
        queues.append(self.fastpulse(name, show, length=8000))
        fastpulsescript, _ = self.combine(queues, seperate_dimmer=False)
        fastpulseid = handler.add_script(name, fastpulsescript, "FastPulse")
        chaserid = handler.add_chaser(name, fastpulseid, "FastPulse", duration=7900)
        handler.add_button(name, "FastPulse", chaserid, "Z")

        queues = []
        queues.append(self.alternate_flood(name, show, length=8000))
        alternatescript, _ = self.combine(queues, seperate_dimmer=False)
        alternateid = handler.add_script(name, alternatescript, "AlternateFlood")
        chaserid = handler.add_chaser(name, alternateid, "AlternateFlood", duration=7900)
        handler.add_button(name, "AlternateFlood", chaserid, "A")

        queues = []
        queues.append(self.side_to_side(name, show, length=8000))
        sidetosidescript, _ = self.combine(queues, seperate_dimmer=False)
        sidetosideid = handler.add_script(name, sidetosidescript, "SideToSide")
        chaserid = handler.add_chaser(name, sidetosideid, "SideToSide", duration=7900)
        handler.add_button(name, "SideToSide", chaserid, "X")

        queues = []
        queues.append(self.slow_flash(name, show, length=8000))
        slowflashscript, _ = self.combine(queues, seperate_dimmer=False)
        slowflashid = handler.add_script(name, slowflashscript, "SlowFlash")
        chaserid = handler.add_chaser(name, slowflashid, "SlowFlash", duration=7900)
        handler.add_button(name, "SlowFlash", chaserid, "C")

            
class Queue:
    def __init__(self):
        self.queue = []

    def get_queue(self):
        return self.queue

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
        self.wav_path = song_data["file"]
        self.beatinterval = 60 / (struct["bpm"])
