# DQN-Autonomous-Vehicle-on-CARLA
An autonomous Vehicle running on CARLA simulator

This project is based on Michael Bosello's repository https://github.com/MichaelBosello/Self-Driving-Car and all his dependencies. Is meant to be a revised version running on CARLA simulator, rather than on real car.

The code is modular, so you can build your own simulated or real car without changing anything else.

## Commands

- standard learning run -> **rl_car_driver.py**
- restore old run -> **rl_car_driver.py --model run-out-xxxx-xx-xx-xx-xx-xx**
- run in evaluation mode -> **rl_car_driver.py --evaluate True --model run-out-xxxx-xx-xx-xx-xx-xx**

## Dependencies

**CARLA simulator requires specific python version to work properly.**

- CARLA 0.9.11
- Python 3.7
- TensorFlow >=2.0,<3.0
- numpy ==1.22.0
- opencv-python
- traitlets
- blosc

Other python versions could work but this one suggested.  
I also suggest to use anaconda.  
At this link the tensorflow installation https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/  
If your system has an NVIDIA gpu, install the gpu version, else you can use the cpu one but it can be much slower.

You can find CARLA simulator at this link, choose the specified above version. https://github.com/carla-simulator/carla/releases  

On Windows, copy  CARLA_Sim_Launcher/Launcher.bat into your folder containing the downloaded simulator (location of CarlaUE4.exe).  This is a workaround for the random server side crashes I've experienced.

note: the simulator was set in low quality to avoid performance issues and shadows, making the learning rate faster.

## Hardware used

This code was tested with 2 pc interconnected with 1Gb ethernet.
A considerable amount of processing power was taken by the raw video stream used.

## Demo

https://youtu.be/5uyfd-dBesE
