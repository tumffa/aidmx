import re
import random
import math
import array
from scipy import interpolate

class ShowStructurer:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.shows = {}
        self.universe_size = data_manager.universe.get("size", 512)
        self.universe = {k: v for k, v in data_manager.universe.items() if k != "size"}

        self.fixture_addresses = {}
        for group in self.universe.values():
            for fixture in group.values():
                self.fixture_addresses[fixture["id"]] = fixture["address"]

        # Dimmer map for help with seperating dimmer commands
        self.fixture_dimmer_map = {}
        for group_name, group in self.universe.items():
            for fixture_key, fixture in group.items():
                fixture_id = fixture["id"]
                dimmer_channel = fixture["dimmer"]
                self.fixture_dimmer_map[fixture_id] = dimmer_channel

        self.dimmer_update_fq = 33 # ms
        # I've observed that QLC+ scripts have compounding lag the longer
        # the script gets, so I add an adjustment to combat this
        self.wait_adjustment = 0.00
        self.pause_wait_adjustment = 0.00
        self.dmx_controller = "ola" # alternatively "qlc"

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
            print("Warning: Empty envelope times or values.")
            return lambda t: 1.0
        
        # Create interpolation function (linear interpolation)
        interp_func = interpolate.interp1d(
            env_times, 
            env_values,
            bounds_error=False,     # Don't raise error for out-of-bounds
            fill_value=(0.01, 0.01)   # Baseline value for out-of-bounds
        )
        
        # Return a closure function that handles the interpolation
        def envelope_function(t):
            return float(interp_func(t))
            
        return envelope_function

    def _setfixture(self, fixture, channel, value, comment=""):
        value = max(0, min(255, value))
        if self.dmx_controller == "qlc":
            return f"fixture:{fixture} channel:{channel} value:{value} //{comment}"
        elif self.dmx_controller == "ola":
            return (fixture, channel, value)

    def _wait(self, time, comment=""):
        time = max(0, int(time))
        if self.dmx_controller == "qlc":
            return f"wait:{int(time)} //{comment}"
        elif self.dmx_controller == "ola":
            return int(time)
    
    def _blackout(self, mode, comment=""):
        if self.dmx_controller == "qlc":
            return f"blackout:{mode} //{comment}"
        elif self.dmx_controller == "ola":
            return 0
    
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

    def pulse(self, name, show, intervalmod=1, dimmer1=255, dimmer2=50, color1="yellow", color2="red", length=30000.0, start=0, queuename="pulse0"):
        result = {} # Dictionary to hold the result: name and queue of commands
        pulse_queue = Queue() # Queue for wait times and commands, which alternate after another.
        # The wait time is an integer and the commands are a list of QLC+ script lines
        result["name"] = queuename
        pulse_queue.enqueue(start) # The first item in the queue is always the time till chaser begins 
        self.shows[name] = show
        group = self.universe["abovewash"] # Define the group of fixtures to use
        time = length # Total time the effect will run for

        # This example sets up a chaser that moves bright color2 to the next fixture every beat
        switchinterval = (show.beatinterval/len(group))*1000*4/intervalmod # Time between switching to the next fixture
        i = 1 # Start with the first fixture
        while time > 1:
            temp = [] # set up a list to hold the commands for this time frame before next wait
            for fixture in group.values():
                # Set fixture to default color
                color_commands = self.calculate_colors(fixture, color1)
                # Add color commands to list
                temp += color_commands
                # Make sure the shutters are open
                temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                # Set the dimmer to dimmer2 (lower brightness)
                temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer2, f"Dimmer reset"))
            # Set fixture i to color2
            color_commands = self.calculate_colors(group[str(i)], color2)
            temp += color_commands
            # Set fixture i to dimmer1 (higher brightness)
            temp.append(self._setfixture(group[str(i)]["id"], group[str(i)]["dimmer"], dimmer1, "Dimmer off"))

            # Decrease the remaining time by the switch interval
            if time - switchinterval < 0:
                switchinterval = time
            time -= switchinterval
            # Move i to the next fixture
            i += 1
            if i > len(group):
                i = 1
            # If there is still time left, enqueue the commands list followed by wait time
            if time > 1:
                pulse_queue.enqueue(temp)
                pulse_queue.enqueue(switchinterval)
        # Append the queue to result
        result["queue"] = pulse_queue
        return result
    
    def fastpulse(self, name, show, intervalmod=4, dimmer1=255, dimmer2=10, color1="all", length=30000.0, start=0, queuename="fastpulse0"):
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
            colors = color1
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
    
    def side_to_side(self, name, show, intervalmod=4, dimmer1=255, dimmer2=255, color1="random", color2=None, length=30000.0, start=0, queuename="sidetoside0"):
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
    
    def randomstrobe(self, name, show, length=30000.0, start=0, queuename="strobe0", strobecolor="white", light_selection_period=50):
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

        # Initialize dimmers 0, shutters closed and colours
        reset = []
        for set in fixtures:
            for fixture in set:
                reset.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                reset.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
                colors = self.calculate_colors(fixture, "white")
                reset += colors
        strobe_queue.enqueue(reset)
        strobe_queue.enqueue(0)
        
        update_period = 10  # ms
        
        while time > 1:
            # Select a new light and color every light_selection_period ms
            if strobecolor == "random":
                color = random.choice(colorspectrum)
                while color == last_color:
                    color = random.choice(colorspectrum)
                last_color = color
                
            # Choose fixture numbers for each set
            active_fixtures = []
            for set_index, set in enumerate(fixtures):
                number = random.randint(0, len(set)-1)
                while number == indexes[set_index]:
                    number = random.randint(0, len(set)-1)
                indexes[set_index] = number
                active_fixtures.append(number)
            
            # Calculate how many update cycles we need in this light selection period
            cycles = min(int(light_selection_period / update_period), int(time / update_period))
            remaining_time = light_selection_period
            
            # Run multiple update cycles with the same active fixture
            for _ in range(cycles):
                temp = []
                for set_index, set in enumerate(fixtures):
                    j = 0
                    for fixture in set:
                        if j == active_fixtures[set_index]:
                            colors = self.calculate_colors(fixture, color)
                            temp += colors
                            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 255, f"Dimmer on"))
                            temp.append(self._setfixture(fixture["id"], fixture["strobe"], fixture["nicestrobe"], f"Strobe on"))
                            fixturedimmers[fixture["id"]] = 255
                        elif fixturedimmers[fixture["id"]] > 0:
                            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], 0, f"Dimmer off"))
                            fixturedimmers[fixture["id"]] = 0
                        j += 1
                
                # Use the shorter update period for more frequent updates
                wait = update_period
                if time < wait:
                    wait = time
                
                strobe_queue.enqueue(temp)
                strobe_queue.enqueue(wait)
                
                time -= wait
                remaining_time -= wait
                
                if time <= 1:
                    break
                    
            # If we still have time but didn't use the full light_selection_period,
            # add a wait for the remaining time
            if time > 1 and remaining_time > 0:
                time -= remaining_time
        
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
            
    def simple_color(self, name, show, dimmer=255, color="red", length=30000.0, start=0, queuename="colorchaser0"):
        """
        Creates a static color effect that sets all lights to the specified color and dimmer
        and maintains that state for the entire duration.
        
        Args:
            name: Show name
            show: Show object
            dimmer: Dimmer value for all fixtures (default 255 - full)
            color: Color to set for all fixtures (default "red")
            length: Length in milliseconds
            start: Start time in milliseconds
            queuename: Name for the queue
        
        Returns:
            Dictionary with queue and name
        """
        result = {}
        color_queue = Queue()
        result["name"] = queuename
        color_queue.enqueue(start)
        self.shows[name] = show
        
        # Use both abovewash and strobe fixtures if available
        group = {}
        if "abovewash" in self.universe:
            group.update(self.universe["abovewash"])
        if "strobe" in self.universe:
            for key, value in self.universe["strobe"].items():
                if key not in group:
                    group[key] = value
        
        # Commands to set color and dimmer
        temp = []
        for fixture in group.values():
            # Set the color
            color_commands = self.calculate_colors(fixture, color)
            temp += color_commands
            
            # Open shutters and set dimmer to full
            temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
            temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer, f"Full dimmer"))
        
        # Add commands and wait for the entire duration
        color_queue.enqueue(temp)
        color_queue.enqueue(length)
            
        result["queue"] = color_queue
        return result
    
    def color_pulse(self, name, show, color1="red", color2="blue", dimmer=255, length=30000.0, start=0, queuename="colorpulse0"):
        """
        Creates a chaser effect that alternates between colors, changing each fixture gradually.
        - Colors use much faster transitions (50 units at a time)
        - Colors always progress forward without going back
        - Ensures pure color transitions (no unintended channel blending)
        """
        result = {}
        colorpulse_queue = Queue()
        result["name"] = queuename
        colorpulse_queue.enqueue(start)  # Start time
        
        # Get fixtures from abovewash group
        fixtures = []
        if "abovewash" in self.universe:
            fixtures = list(self.universe["abovewash"].values())
        else:
            # Return empty result if no abovewash fixtures
            result["queue"] = colorpulse_queue
            return result
        
        # Setup colors
        colorspectrum = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        if color1 == "random":
            color1 = random.choice(colorspectrum)
        if color2 == "random" or color2 is None:
            color2 = random.choice(colorspectrum)
            while color2 == color1:  # Ensure different colors
                color2 = random.choice(colorspectrum)
        
        # Define the color mappings with PURE RGB values - no mixing
        color_map = {
            "white": {"red": 255, "green": 255, "blue": 255},
            "red": {"red": 255, "green": 0, "blue": 0},
            "green": {"red": 0, "green": 255, "blue": 0},
            "blue": {"red": 0, "green": 0, "blue": 255},
            "pink": {"red": 255, "green": 0, "blue": 255},
            "yellow": {"red": 255, "green": 255, "blue": 0},
            "cyan": {"red": 0, "green": 255, "blue": 255},
            "orange": {"red": 255, "green": 80, "blue": 0},
            "purple": {"red": 128, "green": 0, "blue": 255}
        }
        
        # Create a sequence of colors to rotate through
        color_sequence = []
        if color1 == "random" and color2 == "random":
            # Start with 2 different random colors
            current_color = random.choice(colorspectrum)
            color_sequence.append(current_color)
            next_color = random.choice([c for c in colorspectrum if c != current_color])
            color_sequence.append(next_color)
        else:
            # Use the provided colors to start
            color_sequence = [color1, color2]
        
        # Calculate time and steps
        beatinterval = show.beatinterval * 1000  # Beat interval in ms
        time = length
        
        # Track the current position in the color sequence
        current_color_index = 0
        
        # MUCH faster color transitions (50 units at a time)
        color_increment = 50
        
        # Main loop for alternating between color transitions and holds
        while time > 1:
            # Get current and next colors from the sequence
            current_color = color_sequence[current_color_index]
            
            # Calculate next color index (always progress forward)
            next_color_index = (current_color_index + 1) % len(color_sequence)
            next_color = color_sequence[next_color_index]
            
            # Get color values
            source_color_values = color_map[current_color]
            target_color_values = color_map[next_color]
            
            # PHASE 1: TRANSITION to next color
            # Calculate how many steps needed for each color component with larger increments
            steps_needed = {}
            max_steps = 0
            for channel in ["red", "green", "blue"]:
                change = abs(target_color_values[channel] - source_color_values[channel])
                steps_needed[channel] = (change + color_increment - 1) // color_increment  # Ceiling division
                max_steps = max(max_steps, steps_needed[channel])
            
            # Ensure at least 1 step even if no color change
            max_steps = max(1, max_steps)
            
            # Calculate step interval to complete the transition within a beat
            step_interval = beatinterval / max_steps
            
            # Perform the color transition
            current_colors = source_color_values.copy()
            for step in range(max_steps):
                temp = []
                
                # Update each RGB component with larger increments
                for channel in ["red", "green", "blue"]:
                    if step < steps_needed[channel]:
                        direction = 1 if target_color_values[channel] > source_color_values[channel] else -1
                        change = min(color_increment, abs(target_color_values[channel] - current_colors[channel]))
                        current_colors[channel] += direction * change
                    # Explicitly set channels that don't need to change to their target value
                    # This ensures we don't get unwanted blending
                    elif source_color_values[channel] == target_color_values[channel]:
                        current_colors[channel] = target_color_values[channel]
                
                # Apply the current colors to all fixtures
                for fixture in fixtures:
                    if fixture["colortype"] == "seperate":
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], current_colors["red"], f"Red: {current_colors['red']}"))
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], current_colors["green"], f"Green: {current_colors['green']}"))
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], current_colors["blue"], f"Blue: {current_colors['blue']}"))
                    
                    # Set dimmer and shutter
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer, f"Dimmer"))
                    temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                
                colorpulse_queue.enqueue(temp)
                
                # Calculate wait time
                wait_time = step_interval
                if time - wait_time < 0:
                    wait_time = time
                time -= wait_time
                
                if time > 1:
                    colorpulse_queue.enqueue(wait_time)
                else:
                    break
            
            # PHASE 2: HOLD the new color - ensure we set EXACT target color values
            if time > 1:
                temp = []
                # Apply target color to all fixtures - with EXACT color values
                for fixture in fixtures:
                    if fixture["colortype"] == "seperate":
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["red"], target_color_values["red"], f"Hold Red: {target_color_values['red']}"))
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["green"], target_color_values["green"], f"Hold Green: {target_color_values['green']}"))
                        temp.append(self._setfixture(fixture["id"], fixture["colorchannels"]["blue"], target_color_values["blue"], f"Hold Blue: {target_color_values['blue']}"))
                    
                    # Set dimmer and shutter
                    temp.append(self._setfixture(fixture["id"], fixture["dimmer"], dimmer, f"Dimmer"))
                    temp.append(self._setfixture(fixture["id"], fixture["shutter"], fixture["shutters"]["open"], f"Open shutters"))
                
                colorpulse_queue.enqueue(temp)
                
                # Hold for a full beat
                wait_time = beatinterval
                if time - wait_time < 0:
                    wait_time = time
                time -= wait_time
                
                if time > 1:
                    colorpulse_queue.enqueue(wait_time)
                else:
                    break
            
            # Move to the next color in the sequence
            current_color_index = next_color_index
            
            # Every 2 cycles, add a new color to keep things interesting
            if current_color_index % 2 == 0 and color1 == "random":
                # Pick a new color that's different from the last two
                colors_to_avoid = [color_sequence[current_color_index], color_sequence[(current_color_index + 1) % len(color_sequence)]]
                
                if len(colorspectrum) > len(colors_to_avoid):
                    new_color = random.choice([c for c in colorspectrum if c not in colors_to_avoid])
                    
                    # If we've built up more than 3 colors in our sequence, replace the oldest
                    if len(color_sequence) > 3:
                        # Replace the color that's 2 ahead of current (will be used in 2 cycles)
                        replace_index = (current_color_index + 2) % len(color_sequence)
                        color_sequence[replace_index] = new_color
                    else:
                        # Otherwise just add the new color
                        color_sequence.append(new_color)
        
        result["queue"] = colorpulse_queue
        return result
    
    #### this is the meat and potatoes here ####
    def combine(self, queues, end_time=None, seperate_dimmer=True, light_strength_envelope=None, strobe_ranges=None):
        """
        Combines multiple fixture/wait command queues into a single sequence per time segment.
        If needed, keeps track of fixture dimmer values separately and applies envelope scaling.

        Args:
            queues: List of queue dictionaries [wait, [command1, command2, ...], wait, ...]
            start_time: Start of the segment -- time offset from the beginning of the song
            seperate_dimmer: If True, separate dimmer commands from main commands
            light_strength_envelope: Envelope for scaling dimmer values (based on beat and snare/kick pattern)
            strobe_ranges: List of tuples defining strobe ranges (start, end) in milliseconds
            
        Returns:
            Tuple of (main_segment, dimmer_segment)
            All different queues commands in main_segment\n
            Dimmer commands in dimmer_segment if seperate_dimmer is True, otherwise None.\n
            The lists consist of QLC+ commands for a script
        """

        # Check if envelope is provided and initialize the envelope function
        if light_strength_envelope:
            envelope_function = self._light_strength_envelope_function(light_strength_envelope)
        else:
            envelope_function = lambda t: 1.0  # No scaling if no envelope is provided

        # Main pattern commands (excluding dimmer commands)
        segment = []

        # Separate list for dimmer commands
        segment_dimmers = []
        
        # Keep track of fixture states for future restoration after a strobe range
        fixture_dimmers = {}  # {fixture_id: {"channel": channel, "original_value": value}}
        
        # Initialize time tracking (relative to the start of the segment)
        if end_time is None:
            end_time = float('inf')
        current_time_ms = 0
        current_time_sec = current_time_ms / 1000.0
        
        # Update frequency for dimmer scaling during frequent updates
        update_frequency_ms = self.dimmer_update_fq
        
        # Check current script type
        is_slowflash = False
        is_pause = False

        # Initialize command_queues based on provided chaser queues
        command_queues = {}
        for queue in queues:
            command_queues[queue["name"]] = queue["queue"]
            if "slowflash" in queue["name"]:
                is_slowflash = True
            if "pause" in queue["name"]:
                is_pause = True
        
        # Dictionary to keep track of the next wait time for each queue
        times = {}
        for queue in queues:
            times[queue["name"]] = queue["queue"].dequeue()
        
        index = 0
        pattern_started = False
        
        while len(times) > 0:
            do_not_execute = []
            min_time = min(times.values()) # Find the minimum wait time across all queues
            # Find the queue(s) with the same minimum wait time
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
            
            # Choose the queue with the smallest wait time
            queue = command_queues[q[0]]
            wait_time = q[1] # extract the wait time from the queue

            # If this is the first iteration, we need to wait before starting the pattern
            # The first wait time matches segment start time. Wait this time - 1 ms
            # Start with each queue having wait time of 1 ms
            if not pattern_started:
                wait_time -= 1
                segment.append(self._wait(wait_time, f"Initial wait of {wait_time + 1} - 1 ms"))
                if seperate_dimmer:
                    segment_dimmers.append(self._wait(wait_time, f"Initial wait of {wait_time + 1} - 1 ms (dimmer)"))
                for name in times:
                    times[name] -= wait_time # reduce each queue wait to 1
                pattern_started = True
                continue

            # Check if we are entering/in/exiting a strobe range
            entering_strobe_range, in_strobe_range, exiting_strobe_range, wait_time_till_strobe, strobe_range = self.check_strobe_ranges(
                current_time_ms, wait_time, strobe_ranges
            )

            if entering_strobe_range:
                # Adjust wait time to the time till the strobe range start
                wait_time = wait_time_till_strobe

            if in_strobe_range and exiting_strobe_range:
                # Wait till strobe range end + 1 ms
                wait_time = wait_time_till_strobe + 1
            
            # Check if we need to insert dynamic dimmer updates during this wait
            dimmer_condition = (
                wait_time > update_frequency_ms and
                light_strength_envelope and
                fixture_dimmers and
                seperate_dimmer and
                not in_strobe_range and
                not is_pause
            )

            # Initialize variable to check if dimmer updates were found
            found_updates = False

            # If needed, udpdate dimmers script:
            if dimmer_condition:
                active_ranges = light_strength_envelope.get("active_ranges", []) # ranges where the envelope is active
                # Add dynamic dimmer updates during this wait period
                found_updates = self._add_dynamic_dimmer_updates(
                    segment_dimmers, fixture_dimmers, 
                    wait_time, current_time_ms, update_frequency_ms, 
                    envelope_function=envelope_function,
                    active_ranges=active_ranges
                )

            # Add comments for debugging in QLC+
            if entering_strobe_range:
                comment = f"Time: {current_time_sec}. Entering range ({strobe_range[0]} - {strobe_range[1]})"
            elif in_strobe_range:
                comment = f"Time: {current_time_sec}. In range ({strobe_range[0]} - {strobe_range[1]})"
            elif exiting_strobe_range:
                comment = f"Time: {current_time_sec}. Exiting range ({strobe_range[0]} - {strobe_range[1]})"
            else:
                comment = f"Time: {current_time_sec}."

            # Add the remaining wait time to the script
            segment.append(self._wait(wait_time, comment))

            # If no dimmer updates were needed / or condition not met, add same wait time for dimmer script
            if seperate_dimmer and not found_updates:
                segment_dimmers.append(self._wait(wait_time, comment))
            
            current_time_ms += wait_time # Update current time
            current_time_sec = current_time_ms / 1000.0
            
            # Reduce wait time from all script queues
            for name in times:
                times[name] -= wait_time
                if times[name] < 0:
                    times[name] = 0

            # If we are to enter a strobe range before the next command (entering_strobe_range),
            # we can skip processing commands
            if entering_strobe_range:
                continue

            # If we are exiting a strobe range, we need to restore fixture states
            if exiting_strobe_range:
                segment, segment_dimmers = self.restore_fixture_states(
                    fixture_dimmers, segment, segment_dimmers, current_time_sec, envelope_function
                )
                continue
                    
            #### PROCESS COMMANDS ####

            # get all fixture update commands that need to be executed at this timeframe 
            commands = queue.dequeue()

            # if queue is over, remove it
            if commands == None:
                del times[q[0]]
                del command_queues[q[0]]
                continue
            
            if q[0] not in do_not_execute:
                for command in commands:
                    match = False
                    # Extract fixture ID, channel and value from the QLC+ / OLA command
                    if self.dmx_controller == "qlc":
                        match = re.search(r'setfixture:(\d+) ch:(\d+) val:(\d+)', command)
                        fixture_id, channel, value = match.groups()
                        fixture_id = int(fixture_id)
                        channel = int(channel)
                        original_value = int(value)

                    elif self.dmx_controller == "ola" and isinstance(command, tuple):
                        fixture_id, channel, original_value = command
                        match = True

                    if match:
                        # Store this fixture state for later restoration
                        if fixture_id not in fixture_dimmers:
                            fixture_dimmers[fixture_id] = {}
                        fixture_dimmers[fixture_id][channel] = original_value
                        
                        # Only add commands to script if not in a strobe range
                        if not in_strobe_range:
                            # If command is a dimmer command and we are using envelope, we need to scale it with the envelope
                            is_dimmer_command = (fixture_id in self.fixture_dimmer_map and 
                                            channel == self.fixture_dimmer_map[fixture_id])
                            
                            if is_dimmer_command and seperate_dimmer:
                                # Get current envelope strength
                                strength = envelope_function(current_time_sec)
                                
                                if is_slowflash:
                                    # In the case of slowflash chaser, we don't scale the value
                                    scaled_value = original_value
                                else:
                                    scaled_value = int(original_value * strength)
                                
                                # Add dimmer command to the script
                                dimmer_command = self._setfixture(fixture_id, channel, scaled_value,
                                                                f"Scaled value {scaled_value} from {original_value}")
                                segment_dimmers.append(dimmer_command)
                            else:
                                # if not dimmer command, just add the command as is
                                segment.append(command)
                    elif not in_strobe_range:
                        # realistically we should not get here, but if we do, just add the command
                        segment.append(command)

            # Get next wait period
            wait = queue.dequeue()
            if wait:
                times[q[0]] = wait
        
        return segment, segment_dimmers

    def check_strobe_ranges(self, current_time_ms, wait_time, strobe_ranges):
        """ Checks if we are entering, in, or exiting a strobe range.

        Args:
            current_time_ms (_type_): Current time relative to segment start
            wait_time (_type_): Current next wait time in milliseconds
            strobe_ranges (_type_): List of tuples defining strobe ranges (start, end) in seconds

        Returns:
            Tuple of (entering_strobe_range, in_strobe_range, exiting_strobe_range, wait_time_till_strobe, strobe_range)
        """
        entering_strobe_range = False
        in_strobe_range = False
        exiting_strobe_range = False
        wait_time_till_strobe = wait_time
        strobe_range = None

        # Check if we are in a strobe range
        if strobe_ranges:
            current_time_sec = current_time_ms / 1000.0
            end_time_sec = current_time_sec + wait_time /1000.0
            
            for strobe_range in strobe_ranges:
                if strobe_range[0] <= current_time_sec <= strobe_range[1]:
                    in_strobe_range = True
                    break

            # Check if we are entering a strobe range
            if not in_strobe_range:
                for strobe_range in strobe_ranges:
                    if current_time_sec < strobe_range[0] < end_time_sec:
                        entering_strobe_range = True
                        wait_time_till_strobe = (strobe_range[0] - current_time_sec) * 1000
                        break

            # Check if we are exiting a strobe range
            if in_strobe_range:
                for strobe_range in strobe_ranges:
                    if current_time_sec < strobe_range[1] < end_time_sec:
                        exiting_strobe_range = True
                        wait_time_till_strobe = (strobe_range[1] - current_time_sec) * 1000
                        break

        return entering_strobe_range, in_strobe_range, exiting_strobe_range, wait_time_till_strobe, strobe_range

    def restore_fixture_states(self, fixture_dimmers, segment, segment_dimmers, current_time_sec, envelope_function):
        """
        Restores the fixture dimmer states to their original values. This is done when exiting a strobe range.
        
        Args:
            fixture_dimmers: Dictionary of current fixture dimmer values
            segment_dimmers: Dimmer command segment

        Returns:
            Tuple of (segment, segment_dimmers) with commands for restored dimmer values
        """
        for fixture_id, info in fixture_dimmers.items():
            for channel, value in info.items():
                if channel == self.fixture_dimmer_map.get(fixture_id, None):
                    value = value * envelope_function(current_time_sec)
                    dimmer_command = self._setfixture(fixture_id, channel, value, 
                                                        f"Restoring original value {value}")
                    segment_dimmers.append(dimmer_command)
                else:
                    cmd = self._setfixture(fixture_id, channel, value)
                    segment.append(cmd)
        return segment, segment_dimmers

    def _add_dynamic_dimmer_updates(self, segment_dimmers, fixture_dimmers, 
                            wait_time, current_time_ms, update_frequency_ms,
                            envelope_function, active_ranges=None):
        """
        Adds dynamic dimmer updates during long wait periods based on envelope values.
        Uses pre-calculated active ranges that contain a deviation in dimmer strength.
        Handles strobe ranges by stopping updates at strobe boundaries.
        
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
                    segment_dimmers.append(self._wait(wait_between_ranges, f"Wait between ranges"))
                    update_time_ms += wait_between_ranges
                    remaining_wait -= wait_between_ranges
        
        # If we have remaining wait time after all ranges, add final wait
        if remaining_wait > 0:
            segment_dimmers.append(self._wait(remaining_wait, f"Finished, wait for {remaining_wait}ms"))
        return True
    
    def preprocess_onset_ranges(self, start_time, end_time, onset_ranges):
        """
        Preprocess onset ranges for a specific time segment.
        Onset times are provided relative to the song start time, rather than the segment start time.
        
        Args:
            start_time: Start time of segment in seconds
            end_time: End time of segment in seconds
            onset_ranges: List of [start, end] ranges in seconds
            
        Returns:
            List of onset ranges adjusted relative to segments start_time
        """
        if not onset_ranges:
            return []
            
        processed_ranges = []
        segment_duration = end_time - start_time
        
        for onset_start, onset_end in onset_ranges:
            # Check if this onset range overlaps with our segment
            if onset_end <= start_time or onset_start >= end_time:
                # No overlap, skip this range
                continue
                
            # Adjust start if needed
            if onset_start < start_time:
                # If onset starts before our segment, adjust to 0.1s after segment start
                adjusted_start = 0.1
            else:
                # Otherwise shift relative to segment start
                adjusted_start = onset_start - start_time
                
            # Adjust end if needed
            if onset_end > end_time:
                # If onset ends after our segment, adjust to 0.1s before segment end
                adjusted_end = segment_duration - 0.1
            else:
                # Otherwise shift relative to segment start
                adjusted_end = onset_end - start_time
                
            # Make sure we don't have an invalid range after adjustments
            if adjusted_end <= adjusted_start:
                continue
                
            processed_ranges.append([adjusted_start, adjusted_end])
        
        return processed_ranges

    def generate_show(self, name, qxw, delay, strobes=False, simple=False):
        """
            Generates all functions, buttons, etc. for a QLC+ file. 
        Args:
            name (str): Name of the show to generate
            qxw (Object): object that handles qlc+ file generation 
            strobes (bool, optional): Whether or not to include strobe effects. Defaults to False.
            delay (int, optional): Delay in milliseconds to sync lights with music. Defaults to 500
            simple (bool, optional): Whether to use simple mode (only a simple color chaser). Defaults to False.
        """
        print(f"----Combining scripts for show")
        # delay for powershell command to sync song and lightshow start
        scripts = [] # list to hold all of the scripts for file
        function_names = [] # list that holds names of beforementioned scripts
        show = self.create_show(name) # holds information about song
        sections = show.struct["chorus_sections"] # energetic segments of the song
        segments = show.struct["segments"] # segments of the song

        # choose primary chasers and colours - select JUST ONE primary chaser for the whole song
        strong_chasers = ["FastPulse", "SideToSide", "ColorPulse"]
        idle_chasers = ["SimpleColor"]
        primary_chaser = random.choice(strong_chasers)  # Select ONE primary chaser for all energetic segments
        secondary_chaser = random.choice(idle_chasers)
        
        # Choose colors with same logic as before
        colours = ["red", "green", "blue", "pink", "yellow", "cyan", "orange", "purple"]
        
        # Define which colors are visually similar to avoid selecting them together
        close_colors = {
            "blue": ["purple", "cyan"],
            "pink": ["purple"],
            "yellow": ["orange"],
            "cyan": ["blue"],
            "orange": ["yellow"],
            "purple": ["blue", "pink"],
            "red": [],
            "green": []
        }
        
        # Choose first primary color randomly
        primary_color1 = random.choice(colours)
        
        # Choose second primary color that isn't close to the first
        available_colors = [c for c in colours if c != primary_color1 and c not in close_colors[primary_color1]]
        primary_color2 = random.choice(available_colors)
        
        # Primary colors for the energetic chaser
        primary_colours = [primary_color1, primary_color2]
        
        # Choose idle color that isn't close to either primary color
        idle_color_options = [c for c in colours if c not in primary_colours 
                        and c not in close_colors[primary_color1]
                        and c not in close_colors[primary_color2]]
        
        # If no "distant" colors left, just pick one that's not already used
        if not idle_color_options:
            idle_color_options = [c for c in colours if c not in primary_colours]
        
        idle_colour = random.choice(idle_color_options)
        
        # Add premade chasers to use with virtual console
        self.add_chasers(name, show, qxw)

        # Add strobe scripts
        onset_parts = None
        if strobes:
            onset_parts = show.struct["onset_parts"]
            for part in onset_parts:
                queues = []
                queues.append(self.randomstrobe(name, show, length=part[1]*1000-part[0]*1000, start=part[0]*1000 + delay, queuename=f"strobe{part[0]}", light_selection_period=50))
                scripts.append(self.combine(queues)[0])
                function_names.append(f"strobe{part[0]}")

        # Add scripts for each pause (blackout) in the song
        queues = []
        pauses = show.struct["silent_ranges"]
        if self.dmx_controller == "qlc":
            for pause in pauses:
                pause_start = pause[0] / 43
                pause_end = pause[1] / 43
                pausename = f"pause{str(pause[0])[:5]}"
                queues.append(self.pause((pause_end - pause_start), type="blackout", queuename=pausename, start=pause_start*1000 + delay))
            scripts.append(self.combine(queues)[0])
            function_names.append("pauses")

        # Add scripts for each segment in the song
        i = 0
        if segments[0]["label"] == "start":
            i += 1
        onefocus = (len(show.struct["focus"]) == 1) # check how many segments are energetic i.e. verse/chorus/inst

        for i in range(i, len(segments)):
            start_time = segments[i]["start"]*1000 + delay
            end_time = segments[i]["end"]*1000 + delay
            length = (segments[i]["end"] - segments[i]["start"])*1000
            queues = []
            found = False

            for section in sections:
                if segments[i]["start"] == section["seg_start"]:
                    found = True
                    
                    # Use the single primary chaser for all energetic segments
                    current_chaser = primary_chaser
                    is_focus_segment = segments[i]["label"] == show.struct["focus"]["first"]
                    
                    # Override chaser selection for focus segments if needed
                    if not onefocus and is_focus_segment and current_chaser == "ColorPulse":
                        # If it's a focus segment and onefocus is False, don't use ColorPulse
                        current_chaser = random.choice(["FastPulse", "SideToSide"])
                    
                    # Apply the selected chaser
                    if current_chaser == "ColorPulse" or simple == True: # simple mode uses only ColorPulse chaser
                        queues.append(self.color_pulse(
                            name, show, color1=primary_color1, color2=primary_color2, dimmer=255,
                            length=length, start=start_time, queuename=f"colorpulse{i}"))
                    elif current_chaser == "FastPulse":
                        queues.append(self.fastpulse(
                            name, show, color1=[primary_color1, primary_color2],
                            length=length, start=start_time, queuename=f"fastpulse{i}"))
                    elif current_chaser == "SideToSide":
                        queues.append(self.side_to_side(
                            name, show, color1=primary_color1, color2=primary_color2,
                            length=length, start=start_time, queuename=f"sidetoside{i}"))
                    break

            if not found:
                queues.append(self.simple_color(
                    name, show, color=idle_colour, dimmer=255, length=length, 
                    start=start_time, queuename=f"color{i}"))

            if strobes and onset_parts:
                strobe_ranges = self.preprocess_onset_ranges(segments[i]["start"], segments[i]["end"], onset_parts)
            light_strength_envelope = segments[i]["drum_analysis"]["light_strength_envelope"]
            segment_queue, segment_dimmers = self.combine(
                queues,
                end_time=end_time,
                light_strength_envelope=light_strength_envelope,
                strobe_ranges=strobe_ranges if strobes else None,
            )
            scripts.append(segment_queue)
            scripts.append(segment_dimmers)
            # print first 10 entries of dimmer script for debugging
            function_names.append(str(segments[i]["start"]))
            function_names.append(str(segments[i]["start"]) + "_dimmers")
            i += 1

        if self.dmx_controller == "ola":
            result = self.combine_scripts_to_ola_format(scripts)
            return result
        
        result = (scripts, function_names)
        return result
    
    def combine_scripts_to_ola_format(self, scripts):
        """
        Merge multiple scripts (each: [wait_ms, (fixture_id, rel_channel, value), ..., wait_ms, ...])
        into (frame_delays_ms, dmx_frames) for OLA.
        Maps channel to absolute: abs_channel = self.fixture_addresses[fixture_id] + rel_channel
        """
        # Default tick ~30 Hz => minimum ~33 ms between scheduled frames
        max_freq = 30.0
        min_frame_interval_ms = 33 # int(round(1000.0 / max_freq))

        def _is_command_tuple(x):
            return isinstance(x, tuple) and len(x) == 3

        def script_event_iterator(script):
            idx = 0
            n = len(script)
            while idx < n:
                entry = script[idx]
                if not isinstance(entry, int):
                    idx += 1
                    continue

                wait_ms = max(0, entry)
                idx += 1

                commands = []
                while idx < n and not isinstance(script[idx], int):
                    cmd = script[idx]
                    if _is_command_tuple(cmd):
                        commands.append(cmd)
                    elif isinstance(cmd, list):
                        for sub in cmd:
                            if _is_command_tuple(sub):
                                commands.append(sub)
                    idx += 1

                yield wait_ms, commands

        all_scripts = [s for s in (scripts or []) if isinstance(s, list) and s]

        script_states = []
        for script in all_scripts:
            it = script_event_iterator(script)
            try:
                wait_ms, commands = next(it)
            except StopIteration:
                continue
            script_states.append({"iter": it, "next_wait": wait_ms, "pending": commands})

        if not script_states:
            return [0], [array.array("B", [0] * self.universe_size)]

        current_levels = [0] * self.universe_size
        dmx_frames = []
        frame_delays_ms = []

        while script_states:
            min_wait = min(s["next_wait"] for s in script_states)

            if min_wait == 0:
                ready = [s for s in script_states if s["next_wait"] == 0]
                for state in ready:
                    for fixture_id, rel_channel, value in state["pending"]:
                        # Map to absolute channel
                        abs_channel = self.fixture_addresses[fixture_id] + rel_channel
                        # abs_channel is 1-based, so subtract 1 for 0-based index
                        if isinstance(abs_channel, int) and 1 <= abs_channel <= self.universe_size:
                            current_levels[abs_channel - 1] = max(0, min(255, int(value)))
                    try:
                        wait_ms, commands = next(state["iter"])
                        state["next_wait"] = max(0, int(wait_ms))
                        state["pending"] = commands
                    except StopIteration:
                        script_states.remove(state)
                continue

            advance_ms = max(min_wait, min_frame_interval_ms)
            dmx_frames.append(array.array("B", current_levels))
            frame_delays_ms.append(int(advance_ms))

            for state in script_states:
                state["next_wait"] = max(0, state["next_wait"] - advance_ms)

        blackout = array.array("B", [0] * self.universe_size)
        dmx_frames.append(blackout)
        frame_delays_ms.append(min_frame_interval_ms)

        return (frame_delays_ms, dmx_frames)

    def add_chasers(self, name, show, handler):
        return # Disabled for now
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
