import numpy as np
import os
import random
import time
import cv2 as cv
from threading import Thread

from state import State

from car.car_carla import Car
from car.carla_camera import CarlaCamera

MAX_STOP = 3


class CarEnv:
    
    def __init__(self, args, outputDir):
        
        self.outputDir = outputDir

        self.car = Car()
        self.camera = CarlaCamera(self.car)

        #self.motor = CarMotor(CAR_INSTANCE) #rimuovere!!!!!!!!!!!!!!!
        #self.sensor = CarSensor(CAR_INSTANCE) #rimuovere!!!!!!!!!!!!!!!

        self.step_frames = args.frame

        #self.camera.start_recording()

        #self.sensor.add_callback_to_crash(self.stop_callback)

        self.above_line_lx = False
        #self.sensor.add_callback_to_lx_line_sensor(self.lx_line_sensor_callback)
        self.above_line_rx = False
        #self.sensor.add_callback_to_rx_line_sensor(self.rx_line_sensor_callback)

        self.actionSet = [0, 1, 2, 3]
        self.gameNumber = 0
        self.stepNumber = 0
        self.gameScore = 0
        self.episodeStepNumber = 0
        self.frame_number = 0
        self.car_stop_count = 0
        self.prev_action = None
        
        self.isTerminal = False

        self.resetGame()

    def stop_callback(self, channel):
          #self.motor.stop()
          #self.camera.add_note_to_video(self.sensor.get_channel_label(channel))
          time.sleep(0.1)

    def lx_line_sensor_callback(self, channel):
          self.above_line_lx = True
    def rx_line_sensor_callback(self, channel):
          self.above_line_rx = True

    def step(self, action):
        self.isTerminal = False
        self.stepNumber += 1
        self.episodeStepNumber += 1

        for i in range(0, self.step_frames):
          self.frame_number += 1
          self.episode_frame_number +=1
          #self.inputImage = self.camera.capture_as_rgb_array_bottom_half()
          self.inputImage = self.camera.read()

          if self.car.has_crashed():
            #print("crash detected")
            self._resetCar()
            reward = -1
            self.isTerminal = True
            self.car_stop_count = 0
            self.prev_action = None
            self.state = self.state.stateByAddingScreen(self.inputImage.copy(), self.frame_number)
            return reward, self.state, self.isTerminal

          prevScreenRGB = self.inputImage.copy()

          if self.car_stop_count >= MAX_STOP * self.step_frames:
            action = 1

          reward = 0

          if action == 0:
            self.car.action_by_id(0)  #Stop
            reward = -0.01
            self.car_stop_count += 1
          elif action == 1: #Forward
            self.car.action_by_id(1)  #Forward
            reward = 0.5
          elif action == 2: #Left
            self.car.action_by_id(2)  #Left
            reward = 0.3
            if self.prev_action == 2:
              reward = -0.1
          elif action == 3: #Right
            self.car.action_by_id(3)  #Right
            reward = 0.3
            if self.prev_action == 1:
              reward = -0.1
          #elif action == 4:
          #  self.car.action_by_id(4)  #Backward
          #  self.camera.add_note_to_video("action_backward")
          #  reward = -0.6
          else:
            raise ValueError('`action` should be between 0 and 3.')

          if action != 0:
            self.car_stop_count = 0
          
          screenRGB = self.inputImage.copy()

        self.state = self.state.stateByAddingScreen(screenRGB, self.frame_number)
        self.gameScore += reward
        self.prev_action = action
        return reward, self.state, self.isTerminal

    def resetGame(self):
        if self.isTerminal:
            self.gameNumber += 1
            self.isTerminal = False
        #self.state = State().stateByAddingScreen(self.camera.capture_as_rgb_array_bottom_half(), self.frame_number)
        self.state = State().stateByAddingScreen(self.camera.read(), self.frame_number)
        self.gameScore = 0
        self.episodeStepNumber = 0
        self.episode_frame_number = 0
        self.car_stop_count = 0
        self.prev_action = None
        self.car.action_by_id(0)  #Stop


    def stop(self):
      self.car.destroy()
      del self.car

    def _resetCar(self):  #destroy and recreate a new one in a valid position
      while True:
        self.car.destroy()
        del self.car
        del self.camera
        self.car = Car()
        self.camera = CarlaCamera(self.car)
        self.car.apply_control(0.5, -1, 0, False)  #give to it a random movement
        time.sleep(random.randrange(0, 2) + 0.5)
        self.car.apply_control(0, 0, 1, False)
        if not self.car.has_crashed():
          break

    def getNumActions(self):
        return len(self.actionSet)

    def getState(self):
        return self.state
    
    def getGameNumber(self):
        return self.gameNumber
    
    def getFrameNumber(self):
        return self.frame_number
    
    def getEpisodeFrameNumber(self):
        return self.episode_frame_number
    
    def getEpisodeStepNumber(self):
        return self.episodeStepNumber
    
    def getStepNumber(self):
        return self.stepNumber
    
    def getGameScore(self):
        return self.gameScore

    def isGameOver(self):
        return self.isTerminal

        