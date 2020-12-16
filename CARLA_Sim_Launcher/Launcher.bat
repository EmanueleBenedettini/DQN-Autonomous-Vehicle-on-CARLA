@echo off
echo Starting simulator
:start
start /w CarlaUE4.exe
echo Simulator stopped, Restarting...
goto start
