@echo off
echo Starting simulator
:start
start /w CarlaUE4.exe -quality-level=Low
echo Simulator stopped, Restarting...
goto start
