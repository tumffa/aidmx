import DataService
import quiet_before_drop
import StatisticsService
import ShowStructurer
import DMXmanager
import argparse
from pathlib import Path
import os
parser = argparse.ArgumentParser(description="This is my Python script.")

# # Add an argument
# parser.add_argument('arg1', type=str, help='This is argument 1')
# parser.add_argument('arg2', type=str, help='This is argument 2')
# parser.add_argument('arg3', type=str, help='This is argument 3')

# current_filepath = os.path.abspath(__file__)

# args = parser.parse_args()
# name = args.arg1
# file_path = f"{os.path.dirname(current_filepath)}/{args.arg2}"
# write_path = args.arg3

# print(file_path)
# # Get the absolute path of the current file


dm = DataService.DataManager()

def lesgo(name, file):
    dm.extract_data(name, file)
    segments = StatisticsService.segment(name, dm.get_song(name), ["drums", "other"])
    print("------------------SEGMENTS--------------------")
    for segment in segments:
        print(segment)

name = "pause"
file_path = f"./songs/{name}.mp3"
lesgo(name, file_path)

structurer = ShowStructurer.ShowStructurer(dm)
structurer.generate_show(name)


# with open(write_path, 'a') as file:
#     # Write a new line
#     file.write('\nThis is a new line')
# print("done")