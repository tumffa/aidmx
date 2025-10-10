# MP3 to QLC+ Light Show Generator

This project takes an MP3 file and generates a synchronized light show script for QLC+ based on the audio.

DM on discord if you have questions @ ```_tume_```

## Features
- **Segmentation**: Segments the song and applies chasers based on perceived energy.
- **BPM sync**: Chasers are synced to BPM.
- **Dimmer scaling**: Dimmer updates are scaled based on kick/snare hits.
- **Strobes**: Optional strobe effects that are synchronized with drum onsets (most effective with metal music). Currently broken with dimmer scaling.
- **QLC+ Script Generation**:
  - Builds a new QLC file based on an existing fixture template.
  - Writes separate QLC+ dimmer and chaser scripts for each segment, as well as separate scripts for pause blackouts and strobes.
  - Synchronizes all scripts into a QLC+ collection.
  - Adds virtual console buttons for chasers/strobes/blackout

## Showcases
### Kick/snare -based dimmer scaling
[![Kick/snare -based dimmer scaling](https://img.youtube.com/vi/pVIgp4eYaEw/0.jpg)](https://www.youtube.com/watch?v=pVIgp4eYaEw)
### Full song chaser and strobe demo
[![Showcase Video (old version)](https://img.youtube.com/vi/g-IZg1kFES4/0.jpg)](https://youtu.be/g-IZg1kFES4?si=bYKBismXbn0RaHIn)

## Installation

1. Install dependencies from requirements.txt. These dependencies are for `Linux/WSL` with an `Nvidia GPU`. For CPU and other configurations, tweak `requirements.txt`. See [Natten](https://natten.org/install/)  and [allin1](https://github.com/mir-aidj/all-in-one) for more info on how to tweak the installations.
    ```
    pip3 install -r requirements.txt
    ```
2. [Download](https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download) the `LarsNet` inference models, unzip the folder, and place it into `src/services/larsnet/inference_models`.

3. Edit `config.json`:
   ```
   {
      "data_path": "./data",
      "struct_path": "./struct",
      "demix_path": "./demix",
      "setup_path": "./data/Newsetup.qxw", # Your template QLC+ file. The fixtures need to match self.universe of ShowStructurer in showstructurer.py.
      "win_app_path": "/wsl.localhost/Ubuntu-22.04/home/tumffa", # Folder where /aidmx is. This is used for AIQLCshows/play_song.bat Windows script to sync song, not necessary
      "program_data_path": "/mnt/c/ProgramData" # Specify the folder where setup.py will create AIQLCshows folder for generated shows and play_song.bat script will reside.
    }
   ```
4. Copy the template `.qxw` file into `./data` directory.

5. Run `setup.py`. This will create the necessary folders, etc.
