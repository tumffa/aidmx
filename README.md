# MP3 to QLC+ Light Show Generator

This project takes an MP3 file and generates a synchronized light show script for QLC+ based on the audio.

## Dependencies
Making installation easy is still work in progress :)

1. Initialize conda environment from environment.yml.

2. Install **allin1**: https://github.com/mir-aidj/all-in-one?tab=readme-ov-file

3. Rest of dependencies:
```
pip install -r requirements.txt
```

## Features
- **BPM Prediction**: Uses models to estimate the BPM of the song.
- **Segmentation & Instrument Isolation**: Splits the song into segments and separates instrument tracks.
- **Analysis**: Utilizes simple audio analysis with Librosa (e.g., volume) to find the most energetic parts of the song.
- **Strobe Effects**: Adds strobe effects that are synchronized with drum onsets (most effective with metal music).
- **QLC+ Script Generation**: 
  - Writes a separate QLC+ script for each song element.
  - Synchronizes all scripts into a full light show collection.
  - Adds chasers and buttons for additional interactivity.
- **Integration with Existing QLC+ Setup**: Builds a new QLC file based on an existing lighting setup.

## Showcases
### Kick/snare -based dimmer scaling
[![Kick/snare -based dimmer scaling](https://img.youtube.com/vi/pVIgp4eYaEw/0.jpg)](https://www.youtube.com/watch?v=pVIgp4eYaEw)
### Full song chaser and strobe demo
[![Showcase Video (old version)](https://img.youtube.com/vi/g-IZg1kFES4/0.jpg)](https://youtu.be/g-IZg1kFES4?si=bYKBismXbn0RaHIn)

DM on discord if you have questions @ ```_tume_```

Code is still in an experimental stage, so I got some refactoring to do.
