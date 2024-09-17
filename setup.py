import os

# create data/songs,shows,songdata.json
# create struct
# create demix/htdemucs

os.makedirs("data/songs", exist_ok=True)
os.makedirs("data/shows", exist_ok=True)
file = open("data/songdata.json", "w")
file.write("{}")
file.close()
os.makedirs("struct", exist_ok=True)
os.makedirs("demix/htdemucs", exist_ok=True)