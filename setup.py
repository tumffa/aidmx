import os
import shutil
from src.utils import config

config_file = config.get_config("config.json")

# Create the necessary directories and files
data_path = config.get_config("data_path")
struct_path = config.get_config("struct_path")
demix_path = config.get_config("demix_path")
os.makedirs(f"{struct_path}", exist_ok=True)
os.makedirs(f"{demix_path}/htdemucs", exist_ok=True)
os.makedirs(f"{data_path}/songs", exist_ok=True)
os.makedirs(f"{data_path}/shows", exist_ok=True)
file = open(f"{data_path}/songdata.json", "w")
file.write("{}")
file.close()

win_app_path = config.get_config("win_app_path")
song_script_path = "AIQLCshows/play_song.bat"
modified_script_path = "AIQLCshows/play_song_modified.bat"

# Copy the original file to a new file
shutil.copy(song_script_path, modified_script_path)

# Replace APP_PATH with program_data_path in the copied file
with open(modified_script_path, "r") as file:
    data = file.read()
    data = data.replace("APP_PATH", config.to_windows_path(win_app_path))

with open(modified_script_path, "w") as file:
    file.write(data)

# Create the windows shows directory
program_data_path = config.get_config("program_data_path")
os.system(f"mkdir {program_data_path}/AIQLCshows")

# Copy the modified play_song.bat to program_data_path
os.system(f"cp {win_app_path}/{modified_script_path} {program_data_path}/AIQLCshows")