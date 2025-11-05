@echo off
echo ========================================
echo Professional Telemetry Analyzer Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYVER=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% LSS 3 (
    echo ERROR: Python 3.8+ required, found Python %PYVER%
    pause
    exit /b 1
)

if %PYMAJOR% EQU 3 (
    if %PYMINOR% LSS 8 (
        echo ERROR: Python 3.8+ required, found Python %PYVER%
        pause
        exit /b 1
    )
)

echo Python version check: OK
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Could not upgrade pip
) else (
    echo pip upgraded successfully
)
echo.

REM Install dependencies
echo Installing required packages...
echo This may take a few minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To run the application:
echo   python telemetry_analyzer.py
echo.
echo Or double-click: run_analyzer.bat
echo.
pause
