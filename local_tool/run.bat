@echo off
REM Run the tool: run.bat --url "<chapter url>" --output-dir "%USERPROFILE%\Downloads\Manhwa Panels"
cd /d "%~dp0"

if not exist .venv (
    echo No .venv found - run setup.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe cli.py %*
pause
