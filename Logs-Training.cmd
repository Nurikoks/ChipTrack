@echo off
title ChipTRick - Training
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\logs.ps1" -Stream training
