@echo off
:: Set the working directory to the location of the batch file
cd /d "%~dp0"

:: Identify the Python executable in the project Scripts folder
set PYTHON_EXE="%~dp0..\Scripts\python.exe"

:: Check if the Python executable exists
if not exist %PYTHON_EXE% (
    echo [ERROR] Python executable not found at: %PYTHON_EXE%
    echo Ensure this batch file is located in the 'SaaS_Atlas' folder within your project directory.
    pause
    exit /b
)

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [SUCCESS] Running as Administrator.
    echo Starting SaaS Dynamic Atlas Dashboard...
    %PYTHON_EXE% -m streamlit run app.py
) else (
    echo [INFO] Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"cd /d %~dp0 && %~nx0\"' -Verb RunAs"
    exit /b
)
pause