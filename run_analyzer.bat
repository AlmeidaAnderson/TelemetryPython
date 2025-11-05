@echo off
title Professional Telemetry Analyzer

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Launch the analyzer
echo Starting Professional Telemetry Analyzer...
python telemetry_analyzer.py

if errorlevel 1 (
    echo.
    echo ERROR: Application crashed or exited with error
    echo Check that all dependencies are installed: pip install -r requirements.txt
    pause
)
