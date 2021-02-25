import os
import random
import time
import cv2 as cv

from car.car_carla import Car
from car.carla_camera import CarlaCamera
from state import State

MAX_STOP = 5


class CarEnv:

    def __init__(self, args, base_output_dir):

        self.evaluate_run = args.evaluate
        self.car = Car(high_res_capture=self.evaluate_run)
        self.camera = CarlaCamera(self.car)

        self.step_frames = args.history_length 

        self.actionSet = [0, 1, 2, 3]
        self.gameNumber = 0
        self.stepNumber = 0
        self.gameScore = 0
        self.episodeStepNumber = 0
        self.frame_number = 0
        self.car_stop_count = 0
        self.prev_action = None
        self.inputImage = self.camera.read()
        self.show_images = args.show_images
        self.isTerminal = False

        self.base_output_dir = base_output_dir

        self.reset_game()

    def step(self, action, isTraining = False):
        self.isTerminal = False
        self.stepNumber += 1
        self.episodeStepNumber += 1

        for i in range(0, self.step_frames):
            self.frame_number += 1
            self.episode_frame_number += 1
            self.inputImage = self.camera.read()

            if self.car.has_crashed() or self.car.is_server_crashed():
                # print("crash detected")
                reward = -1
                self.isTerminal = True
                self.car_stop_count = 0
                self.prev_action = None
                self.state = self.state.state_by_adding_screen(self.inputImage.copy(), self.frame_number)
                return reward, self.state, self.isTerminal

            prevScreenRGB = self.inputImage.copy()

            #if self.car_stop_count >= MAX_STOP * self.step_frames:
                #action = 1

            reward = 0

            if action == 0:
                self.car.action_by_id(0)  # Stop
                self.car_stop_count += 1
                reward = -0.1
                if self.car.front_side_proximity_detector():
                    reward = 0.1

            elif action == 1:  # Forward
                self.car.action_by_id(1)  # Forward
                reward = 0.7
                if self.car.front_side_proximity_detector():
                    reward = -0.7
                    
            elif action == 2:  # Left
                self.car.action_by_id(2)  # Left
                reward = 0.22
                if self.prev_action == 3:   # if dancing around
                    reward = 0
                if self.car.left_side_proximity_detector():
                    reward = -0.22

            elif action == 3:  # Right
                self.car.action_by_id(3)  # Right
                reward = 0.22
                if self.prev_action == 2:   # if dancing around
                    reward = 0
                if self.car.right_side_proximity_detector():
                    reward = -0.22
            
            else:
                raise ValueError('`action` should be between 0 and 3.')

            if action != 0:
                self.car_stop_count = 0

            screenRGB = self.inputImage.copy()

        self.state = self.state.state_by_adding_screen(screenRGB, self.frame_number)
        self.gameScore += reward
        self.prev_action = action
        return reward, self.state, self.isTerminal

    def reset_game(self):
        self._reset_car()
        self.gameNumber += 1
        self.isTerminal = False
        self.state = State().state_by_adding_screen(self.camera.read(), self.frame_number)
        self.gameScore = 0
        self.episodeStepNumber = 0
        self.episode_frame_number = 0
        self.car_stop_count = 0
        self.prev_action = None

    def stop(self):
        self.car.destroy()
        del self.car

    def _reset_car(self):  # destroy and recreate a new one in a valid position
        if self.evaluate_run:
            self.save_demo_video()

        if self.show_images:
            self.show_demo_video()

        reinitialize = self.episodeStepNumber<10 or self.car.is_server_crashed()
        if reinitialize:
            while True:
                self.car.destroy()
                del self.car
                del self.camera
                self.car = Car(high_res_capture=self.evaluate_run)
                self.camera = CarlaCamera(self.car)
                #self.car.apply_control(0.5, -1, 0, False)  # give it a random movement
                #time.sleep(random.randrange(0, 2) + 0.5)
                #self.car.apply_control(0, 0, 1, False)
                if not self.car.has_crashed():
                    break
        else:
            self.car.reset_position()

    def get_state_size(self):
        return len(self.state.get_screens())

    def get_num_actions(self):
        return len(self.actionSet)

    def get_state(self):
        return self.state

    def get_game_number(self):
        return self.gameNumber

    def get_frame_number(self):
        return self.frame_number

    def get_episode_frame_number(self):
        return self.episode_frame_number

    def get_episode_step_number(self):
        return self.episodeStepNumber

    def get_step_number(self):
        return self.stepNumber

    def get_game_score(self):
        return self.gameScore

    def is_game_over(self):
        return self.isTerminal

    def show_demo_video(self):
        image_list, _ = self.car.get_image_list()
        for img in image_list:
            cv.imshow("Car main camera", img)
            cv.waitKey(30)

    def save_demo_video(self):
        video_dir = self.base_output_dir + '/videos/'
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)
        img_array, size = self.car.get_image_list()
        x, y = size
        zise = (y, x)
        fourcc = cv.VideoWriter_fourcc(*'DIVX') # NOTE this depends on your OS.
        out = cv.VideoWriter(video_dir + str(self.get_game_number()) + "_" + str(round(self.get_game_score(),1)) + 'pts.avi', fourcc, 30, zise)
        for im in img_array:
            out.write(im)
        out.release()