@echo off
title Stop DeepInterview
echo Stopping any running DeepInterview services (ports 3000, 8000, 8880)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 :8000 :8880"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo All services stopped cleanly.
timeout /t 2 >nul
