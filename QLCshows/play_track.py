import sys
import subprocess
import pyautogui
import time

def open_file(path):
    command = f"C:\\QLC+\\qlcplus.exe -f --open {path}"
    subprocess.Popen(command, shell=True)
    # List of images to click on
    images = ['C:/ProgramData/QLCshows/gui/functions.png',
               'C:/ProgramData/QLCshows/gui/collection.png',
                 'C:/ProgramData/QLCshows/gui/newcollection.png',
                 'C:/ProgramData/QLCshows/gui/operate.png',
                   'C:/ProgramData/QLCshows/gui/play.png']
    time.sleep(2)
    for image in images:
        while True:
            # Try to locate the center of the image on the screen
            location = pyautogui.locateCenterOnScreen(image)

            # If the image is found, break the loop
            if location is not None:
                break
            # If the image is not found, wait for a second and try again
            time.sleep(0.5)
        # Double click on 'collection.png', click once on the others
        if image == 'C:/ProgramData/QLCshows/gui/collection.png':
            pyautogui.doubleClick(location)
        else:
            pyautogui.click(location)

commands = sys.argv[1:]
print(commands)

if len(commands) == 0:
    print("No commands given")
    sys.exit(1)

command = commands[0]
filepath = commands[1]

if command == "play":
    subprocess.run("taskkill /IM qlcplus.exe /F", shell=True)
    open_file(filepath)
    sys.exit()