@echo off
if "%~1"=="" (
    echo Usage: run_phone.cmd PHONE_IP
    echo Example: run_phone.cmd 192.0.2.1
    exit /b 1
)
call "%~dp0run.cmd" --camera "http://%~1:4747/video"
