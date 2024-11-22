import json

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_config(param : str):
    config = load_config()
    return config.get(param)

def to_windows_path(path):
    return path.replace("/", "\\\\").replace("\\\\mnt\\\\c\\\\", "C:\\")