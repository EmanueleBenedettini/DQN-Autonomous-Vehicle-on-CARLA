# DQN-Autonomous-Vehicle-on-CARLA
An autonomous Vehicle running on CARLA simulator

This project is based on Michael Bosello's repository https://github.com/MichaelBosello/Self-Driving-Car and all his dependencies. Is meant to be a revised version running on CARLA simulator, rather than on real car.

The code is modular, so you can build your own simulated or real car whitout changing anything else.

## Commands

- standard learning run -> **rl_car_driver.py**
- restore old run -> **rl_car_driver.py --model run-out-xxxx-xx-xx-xx-xx-xx**
- run in evaluation mode -> **rl_car_driver.py --evaluate True --model run-out-xxxx-xx-xx-xx-xx-xx**

## Dependecies

**CARLA simulator requires specific python version to work properly.**

- CARLA 0.9.11
- python 3.7
- TensorFlow 2.x

Other python versions could work but this one suggested.  
I also suggest to use anaconda.  
At this link the tensorflow intallation https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/  
If your system has an NVIDIA gpu, install the gpu version, else you can use the cpu one but it can be much slower.

You can find CARLA simulator at this link, choose the specified above version. https://github.com/carla-simulator/carla/releases  

On Windows, copy  CARLA_Sim_Launcher/Launcher.bat into your folder containig the downloaded simulator (location of CarlaUE4.exe).  This is a workaround for the random server side crashes I've experienced.

note: the simulator was set in low quality to avoid performance issues and shadows, making the learning rate faster.

## Hardware used

This code was tested whit 2 pc interconnected whit 1Gb ethernet.

- Server: AMD FX8350, 16GB 1600MHz, AMD RX580 4GB
- Client: Intel i7-4710HQ, 8GB 1600MHz, NVidia GTX850M 2GB

The pc experiencing the highest load was the server one which was handling all the rendering and pysical work. A considerable amount of processing was also taken by the ethernet communication.

## Demo

https://youtu.be/5uyfd-dBesE
