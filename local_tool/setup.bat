@echo off
REM One-time setup on Windows. Double-click this file, or run: setup.bat
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found.
    echo Install it from https://www.python.org/downloads/ - during install, check
    echo "Add python.exe to PATH". Torch does not yet support the newest Python,
    echo so 3.12 specifically is required.
    pause
    exit /b 1
)

py -3.12 -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt

echo.
echo Setup done. Double-click "Manhwa Downloader.vbs" in this folder to open it -
echo no command prompt needed from here on. (Command-line alternative: run.bat --url "<chapter url>" --output-dir "%USERPROFILE%\Downloads\Manhwa Panels")
pause
