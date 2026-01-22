# MP3 to QLC+ / OLA DMX Show Generator

This project takes an MP3/wav file and generates a synchronized light show script for QLC+ / OLA based on the analysis.

DM on discord if you have questions @ ```_tume_```

## Features
- **Segmentation**: Segments the song and applies chasers based on perceived energy.
- **BPM sync**: Chasers are synced to BPM.
- **Dimmer scaling**: Dimmer brightness can be scaled based on drum beat or volume flow.
- **Strobes**: Optional strobe effects that are synchronized with drum onsets (most effective with metal music).
- **QLC+ Script Generation**:
  - Builds a new QLC file based on an existing fixture template.
  - Writes separate QLC+ dimmer and chaser scripts for each segment, as well as separate scripts for strobes.
  - Synchronizes all scripts into a QLC+ collection.
- **OLA DMX control**:
  - DMX states and execution timeframes are written to a json file
  - OLA can be used to play the generated show and song playback is synced.

## Showcases
### Kick/snare -based dimmer scaling
[![Kick/snare -based dimmer scaling](https://img.youtube.com/vi/pVIgp4eYaEw/0.jpg)](https://www.youtube.com/watch?v=pVIgp4eYaEw)
### Full song chaser and strobe demo
[![Showcase Video (old version)](https://img.youtube.com/vi/g-IZg1kFES4/0.jpg)](https://youtu.be/g-IZg1kFES4?si=bYKBismXbn0RaHIn)

## Installation Linux (Python 3.10)

1. Install PyTorch, Natten, and rest of dependencies:
    - PyTorch:
    ```
    pip install -f https://download.pytorch.org/whl/cu121 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
    ```
    - Natten:
    ```
    # Get python version:
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    
    # Install natten wheel
    pip install "https://github.com/SHI-Labs/NATTEN/releases/download/v0.15.0/natten-0.15.0+torch210cu121-cp${PYVER}-cp${PYVER}-linux_x86_64.whl"
    ```
     - Rest of dependencies:
    ```
    pip3 install -r requirements.txt
    ```
3. Install ola and ola-python (skip if you do not intend to use OLA):
    ```
    sudo apt install ola ola-python
    ```
4. [Download](https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download) the `LarsNet` inference models, unzip the folder, and place it into `src/services/larsnet/inference_models`.

5. Edit `config.json`:
   ```json
   {
      "data_path": "./data",
      "struct_path": "./struct",
      "demix_path": "./demix",
      "setup_path": "./data/Newsetup.qxw", # Your template QLC+ file
      "win_app_path": "/wsl.localhost/Ubuntu-22.04/home/tumffa", # Project folder. This is used for AIQLCshows/play_song.bat Windows script to sync song, not necessary
      "program_data_path": "/mnt/c/ProgramData" # Specify the folder where setup.py will create AIQLCshows folder for generated shows and play_song.bat script will reside.
    }
   ```
6. Copy your template `.qxw` file into `./data` directory.

7. Run `setup.py`. This will create the necessary folders, etc.

## Universe setup and usage

### Universe setup
Configure `universe` in `config.json` to match the configuration in your template `.qxw` file. The chasers that are implemented are mostly based on just a row of washers such as this. Use the `abovewash` -key for such setup.
  ```json
  "universe": {
    "size": 32,
    "abovewash": {
      "1": {"id": 1, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": [20, 255],
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250, "address": 7},
      "2": {"id": 2, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": [20, 255],
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250, "address": 12},
      "3": {"id": 3, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": [20, 255],
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250, "address": 17},
      "4": {"id": 4, "dimmer": 3, "colortype": "seperate", "colorchannels": {"red": 0, "green": 1, "blue": 2}, "strobe": 4, "stroberange": [20, 255],
                    "shutter": 4, "shutters": {"open": 0}, "nicestrobe": 250, "address": 22}
    }
  }
  ```
**Meanings of keys:**
- **"id"**: QLC fixture ID, these should be in physical order
- **"dimmer"**: QLC channel for dimmer
- **"colortype"**: "seperate" for different RGB channels, single channel not supported right now
- **"colorchannels"**: channels for red, green, blue if colortype is seperate
- **"strobe"**: QLC channel for strobe
- **"stroberange"**: range of strobe values that produce a strobe effect
- **"shutter"**: QLC channel for shutter
- **"shutters"**: possible shutter values, "open" is required, "closed" is optional
- **"nicestrobe"**: a strobe value for strobe channel that produces a nice strobe effect
- **"address"**: the address of the fixture in the universe [1-512]

### Usage
**To generate QLC+ / OLA show**, first move the desired `.mp3` / `.wav` file into `./data/songs`.

Then, in the main directory, run:

  ```
  python3 generate_show.py song_name [-st] [-si] [-d] [-l]
  ```
- `-st` can be used to turn strobes effects on
- `-si` can be used to limit the energetic chasers to `color_pulse`, which is good if you're running only a couple fixtures.
- `-d` can be used to define a delay before the start of the QLC+ show. Default is 1 second, if you want to use the PowerShell script.
- `-l` can be used to define a scaling factor for QLC+ dimmer script wait times. This is to help with the compounding lag, scale down/up if beat flashes are too slow/fast.

**To playback the OLA sequence and song**, run:

  ```
  python3 ola_playback.py song_name [-d] [-u] [-s]
  ```
- `-d` can be used to set the delay before DMX starts to sync it with song playback.
- `-u` can be used to set the universe, default 1.
- `-s` can be used to define the start time (in seconds)

Alternatively, for more options through the command-line interface, run:
  ```
  python3 src/main.py
  ```

## Customization
New chasers can be added by hardcoding them into `ShowStructurer` in `showstructurer.py`.
The chaser should return a dictionary:
  ```json
  {
    "name": "name of the chaser script",
    "queue": "Queue() object"
  }
  ```
The `Queue()` object should begin with an `integer` (initial wait time) and then alternate between tuples of `(fixture_id, channel, value, scaling)` and wait time `integers`.
The `combine` function will take any number of these chaser queue dictionaries and alter the wait times so that each chaser can run concurrently.
It then combines them into a dimmer script and a script containing the other commands. Use `self._setfixture` to generate script lines. 
For instance, to set fixture 1 dimmer to 255 and use no scaling, use following:
  ```python
  fixture = self.universe["abovewash"]["1"]
  command = self._setfixture(fixture["id"], fixture["dimmer"], 255)
  ```
Dimmer values can be scaled with beat/flow envelopes. You can still create chaser patterns using dimmers, because the envelope will simply scale the values.
- Scaling with the "beat" envelope will result in the dimmers flashing to the beat.
- Scaling with the "flow" envelope will result in the dimmers to be scaled by the volume of vocals/bass/other.
- Scaling with the "both" envelope results in a mix, where the beat envelope is prioritized, but flow is used when there is no beat.

To scale the dimmer values, use the scale_dimmer parameter:
  ```python
  command = self._setfixture(fixture["id"], fixture["dimmer"], 255, scale_dimmer="both")
  ```

Here is an example of a chaser that moves a bright color2 to the next fixture every beat:
  ```python
    def pulse(self, name, show, intervalmod=1, dimmer1=255, dimmer2=50, color1="yellow", color2="red", length=30000.0, start=0, queuename="pulse0"):
        result = {} # Dictionary to hold the result: name and queue of commands
        pulse_queue = Queue() # Queue for wait times and commands, which alternate after another.
        # The wait time is an integer and the commands are a list of QLC+ script lines
        result["name"] = queuename
        pulse_queue.enqueue(start) # The first item in the queue is always the time till chaser begins 
        self.shows[name] = show
        group = self.universe["abovewash"] # Define the group of fixtures to use
        time = length # Total time the effect will run for

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
  ```

**To change the logic of how chasers are selected**, edit `generate_show` in `showstructurer.py`:

```python
for i in range(i, len(segments)):
    start_time = segments[i]["start"]*1000
    end_time = segments[i]["end"]*1000
    length = (segments[i]["end"] - segments[i]["start"])*1000
    queues = []
    if segments[i]["is_chorus_section"]:
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
    else:
        queues.append(self.simple_color(
            name, show, color=idle_colour, dimmer=255, length=length, 
                    start=start_time, queuename=f"color{i}"))
```
