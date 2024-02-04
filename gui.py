import pyautogui
import time
import pyperclip as pc
import subprocess

wsl_path = open("wsl_filepath.txt", "r").read()
script_path = open("script_path.txt", "r").read()
# print(wsl_path)
# location = pyautogui.locateOnScreen('newscript.png')
# center = pyautogui.center(location)
# pyautogui.click(center)
# fo = open("skripti.txt", "r").read()
# pc.copy(fo)
# location = pyautogui.locateOnScreen('copyfromclipboard.png')
# center = pyautogui.center(location)
# pyautogui.click(center)


def info():
    print("Commands: ")
    print("analyze <name> <file.mp3>")
    print("exit  -  exit the program")

while True:
    info()
    command = input("\nEnter command: ")
    if command == "exit":
        break
    elif command == "analyze":
        name = input("Enter name: ")
        file = input("Enter file: ")
        script = "ServiceHandler.py"
        cmd = f"wsl -e python3 {wsl_path}/{script} {name} {file} {script_path}"
        print(cmd)
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(process.stdout)
    else:
        print("Invalid command")
        info()
        continue