@echo off
title ChipTRick - Versions
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\logs.ps1" -Stream versions
