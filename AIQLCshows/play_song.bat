@echo off
REM Description: Play a song using the provided song name.
REM Syntax: play_song.bat songName

REM Check if the song name is provided
if "%~1"=="" (
    echo Syntax error: Missing song name argument.
    exit /b 1
)

REM Set the song name
set "songName=%~1"

REM Start playing the song using PowerShell in a separate process
start powershell -Command "(New-Object System.Media.SoundPlayer 'APP_PATH\\aidmx\\data\\songs\\%songName%').PlaySync()"

REM Wait for user input to exit
echo Press Ctrl+C to stop playing the song.
pause >nul