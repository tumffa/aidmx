import os  # OS operations
import shutil  # File operations
from src.utils import config  # Custom config module


# Load configuration file
try:
    config_file = config.get_config("config.json")
except Exception as e:
    print(f"Error loading config file: {e}")
    config_file = None


# Create the necessary directories and files
try:
    data_path = config.get_config("data_path")
    struct_path = config.get_config("struct_path")
    demix_path = config.get_config("demix_path")
    os.makedirs(f"{struct_path}", exist_ok=True)
    os.makedirs(f"{demix_path}/htdemucs", exist_ok=True)
    os.makedirs(f"{data_path}/songs", exist_ok=True)
    os.makedirs(f"{data_path}/shows", exist_ok=True)
    print(f"Created struct path: {struct_path}")
    print(f"Created demix path: {demix_path}/htdemucs")
    print(f"Created song path: {data_path}/songs")
    print(f"Created shows path: {data_path}/shows")
    # Create empty songdata.json
    with open(f"{data_path}/songdata.json", "w") as file:
        file.write("{}")
        print(f"Created empty songdata.json at {data_path}/songdata.json")
except Exception as e:
    print(f"Error creating directories or files: {e}")


# Prepare Windows batch script
try:
    win_app_path = config.get_config("win_app_path")
    song_script_path = "AIQLCshows/play_song.bat"
    modified_script_path = "AIQLCshows/play_song_modified.bat"
    # Copy the original file to a new file
    shutil.copy(song_script_path, modified_script_path)
    # Replace APP_PATH with program_data_path in the copied file
    with open(modified_script_path, "r") as file:
        data = file.read()
        project_folder_name = os.path.basename(os.getcwd())
        data = data.replace("APP_PATH", config.to_windows_path(win_app_path) + f"\{project_folder_name}")
    with open(modified_script_path, "w") as file:
        file.write(data)
    print(f"Prepared batch script at {modified_script_path}")
except Exception as e:
    print(f"Error preparing batch script: {e}")


# Create the windows shows directory and copy batch file
try:
    program_data_path = config.get_config("program_data_path")
    # Create directory (ignore error if exists)
    os.system(f"mkdir {program_data_path}/AIQLCshows")
    print(f"Created program data path: {program_data_path}/AIQLCshows")
    # Copy the modified play_song.bat to program_data_path
    os.system(f"cp {modified_script_path} {program_data_path}/AIQLCshows")
    print(f"Copied batch script to {program_data_path}/AIQLCshows")
except Exception as e:
    print(f"Error copying batch file to program data path: {e}")