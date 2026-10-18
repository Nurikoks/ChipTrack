@echo off
title ChipTRick - All streams
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\logs.ps1" -Stream all
