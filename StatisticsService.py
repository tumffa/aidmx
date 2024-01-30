import librosa


def get_rms(song_data=None, category=None, path=None):
    # Load the segment data from the JSON file
    if path:
        data_path = path
    elif not category:
        data_path = song_data['file']
    elif song_data:
        data_path = f"{song_data['demixed']}{category}.wav"
    else:
        raise Exception("Invalid arguments")
    # Calculate the average intensity
    data_y, sr = librosa.load(data_path)
    rms = librosa.feature.rms(y=data_y)[0]
    return rms