@echo off
title ChipTRick - Inference
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\logs.ps1" -Stream inference
