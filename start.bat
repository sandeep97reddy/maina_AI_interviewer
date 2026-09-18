@echo off
title DeepInterview
cd /d "%~dp0"
apps\agent\.venv\Scripts\python.exe scripts\run_all.py
pause
