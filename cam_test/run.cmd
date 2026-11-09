@echo off
"%~dp0..\.venv\Scripts\python.exe" "%~dp0webcam.py" %*
if errorlevel 1 pause
