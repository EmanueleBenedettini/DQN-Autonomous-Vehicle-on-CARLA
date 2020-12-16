# DQN-Autonomous-Vehicle-on-CARLA
An autonomous Vehicle running on CARLA simulator

This project is based on Michael Bosellos's repository https://github.com/MichaelBosello/Self-Driving-Car and is meant to be a revised version running on CARLA simulator, rather than on real car. The code is modular, so building new car instances is very easy!

Now the DQN inputs and outputs are only passed to the car instance, not as before. It is the car instance that orchestrates the data coming from the sensors and aggregates them into a single frame, which is fed to the DQN

## Commands

- standard learning run -> **rl_car_driver.py**
- restore old run -> **rl_car_driver.py --model run-out-xxxx-xx-xx-xx-xx-xx --epsilon YourLastEpsilon**

note: setting epsilon and epsilon-min to 0 makes the code run on eval mode.

## Dependecies

**CARLA simulator requires specific python version to work properly.**

- CARLA 0.9.8
- python 3.7.7
- TensorFlow 2.x (actually used in v1 mode)


You can find CARLA simulator at this link, choose the specified version. https://github.com/carla-simulator/carla/releases  
On Windows, copy  CARLA_Sim_Launcher/Launcher.bat into your folder containig the downloaded simulator (location of CarlaUE4.exe).  This is a workaround for the random server side crashes I've experienced, this script it's just restarting it every time it closes. 
