#!/usr/bin/env python3.7.7
# Note: use Python 3.7.7

import os
import sys
import glob
import random
import cv2 as cv
import numpy as np
import time
import math
from state import State
import threading
#import numba

from utils.config import load_data

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('./PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
                    )
except IndexError:
    print("EGG not found")
pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
# -- end of carla module search ------------------------------------------------


CONFIGFILE = "config.json"


def _action_to_car_values(act):  # 0=stop, 1=forward, 2=left, 3=right
    if act == 0:
        return 0, 0, 1, False  # STOP
    elif act == 1:
        return 1, 0, 0, False  # FORWARD
    elif act == 2:
        return 0.5, -0.5, 0, False  # LEFT
    elif act == 3:
        return 0.5, 0.5, 0, False  # RIGHT
    else:
        raise ValueError('`action` should be between 0 and 3.')

def _action_to_string(act):  # 0=stop, 1=forward, 2=left, 3=right
    if act == 0:
        return "Stop"  # STOP
    elif act == 1:
        return "Forward"  # FORWARD
    elif act == 2:
        return "Left"  # LEFT
    elif act == 3:
        return "Right"  # RIGHT
    else:
        return "err"


class Car:

    def process_img(self, raw_image):
        self.image = np.array(raw_image.raw_data)
        self.image = self.image.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        self.image = cv.cvtColor(self.image, cv.COLOR_RGBA2RGB)
        if self.high_res_capture and self.recording:
            if self.get_speed() > 0.1:
                image_whit_hud = cv.putText(self.image.copy(), str(round(self.get_speed(), 1))+"m/s  "+_action_to_string(self.current_control), (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA) 
                self.image_array.append(image_whit_hud)
        self.image = self.image[int(self.IM_HEIGHT//2.4)::] # trim the top part
        self.image_available = True
        return

    def process_img_semantic(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        i2 = i2[int(self.IM_HEIGHT//2.4)::] # trim the top part
        self.semantic_image = i2[:, :, 2]
        return

    def __init__(self, high_res_capture=False):

        self.high_res_capture = high_res_capture

        # aspect ratio will always be 1 height, 2 width
        if self.high_res_capture:
            self.IM_HEIGHT = 240
        else:
            self.IM_HEIGHT = State.IMAGE_HEIGHT
        self.IM_WIDTH = self.IM_HEIGHT*2

        self.actor_list = []
        self.image = []
        self.image_array = []
        self.image_available = False
        self.semantic_image = []
        self.current_control = 0
        self.current_applied_control = (0, 0, 0, False)
        self.server_crash_detected = False
        self.recording = True
        self.choosen_speed = 0
        self.choosen_sterring = 0
        self.stop = False

        settings = load_data(CONFIGFILE)

        self.initialized = False
        while not self.initialized:
            try:
                connection_data = settings["connection"]
                client = carla.Client(connection_data["server_ip"], connection_data["server_port"])
                client.set_timeout(10.0)

                world = client.get_world()
                blueprint_library = world.get_blueprint_library()

                bp = blueprint_library.filter("model3")[0]  # model3

                spawn_point = random.choice(world.get_map().get_spawn_points())
                self.vehicle = world.spawn_actor(bp, spawn_point)
                self.actor_list.append(self.vehicle)

                # camera initialization
                cam_bp = blueprint_library.find("sensor.camera.rgb")
                cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
                cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
                spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
                self.camera = world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
                self.actor_list.append(self.camera)
                self.camera.listen(lambda data: self.process_img(data))
                # print("camera initialized")

                # semantic camera initialization
                cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
                cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
                cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
                spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
                self.s_camera = world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
                self.actor_list.append(self.s_camera)
                self.s_camera.listen(lambda data: self.process_img_semantic(data))
                # print("semantic camera initialized")

                # collision initialization
                coll_bp = blueprint_library.find("sensor.other.collision")
                self.collision = world.spawn_actor(coll_bp, spawn_point, attach_to=self.vehicle)
                self.actor_list.append(self.collision)
                self.collisions = []
                self.collision.listen(lambda data: self.collisions.append(data))
                # print("collision sensor initialized")

                self.initialized = True

            except RuntimeError:
                print("Init phase failed, check server connection. Retrying in 30s")
                time.sleep(30)
                self.initialized = False
            #End of while

        time.sleep(3)
        self.start_position = self.get_position()
        self.thread = threading.Thread(target=self._body_controller)
        self.thread.start()

    def _body_controller(self):
        while not self.stop:
            steer = self.choosen_sterring
            current_speed = self.get_speed()
            if current_speed < 0.1:
                current_speed = 0.1

            if self.choosen_speed <= 0.1:
                throttle = 0
                brake = 1
            elif current_speed <= self.choosen_speed:
                throttle = max(min((self.choosen_speed/current_speed)-1, 1), 0)
                brake = 0
            else:
                throttle = 0
                brake = max(min((current_speed/self.choosen_speed)-1, 1), 0)
            self._apply_control(throttle, steer, brake, False)
            time.sleep(0.001)

    def _apply_control(self, throttle=0.0, steer=0.0, brake=0.0, reverse=False):  # throttle 0:1, steer -1:1, brake 0:1
        self.current_applied_control = (throttle, steer, brake, reverse)
        self.vehicle.apply_control(
        #                           carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=False, manual_gear_shift=True, gear=1))
                                   carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=False))

    def action_by_id(self, act):
        self.current_control = act
        #throttle, steer, brake, reverse = _action_to_car_values(act)
        #self.apply_control(throttle, steer, brake, reverse)
        if act == 0:
            self.set_speed(0.5)
            #self.set_sterring(0)
        elif act == 1:
            self.set_speed(10)
            self.set_sterring(0)
        elif act == 2 or act == 3:
            if act == 2:
                self.set_sterring(-0.5)
            else:
                self.set_sterring(0.5)
            self.set_speed(2)

    def set_speed(self, speed):
        self.choosen_speed = speed
        if self.choosen_speed < 0:
            self.choosen_speed = 0

    def set_sterring(self, sterring):
        self.choosen_sterring = sterring
        if self.choosen_sterring < (-1):
            self.choosen_sterring = -1
        elif self.choosen_sterring > 1:
            self.choosen_sterring = 1

    def get_im_sizes(self):
        return self.IM_WIDTH, self.IM_HEIGHT

    def get_image(self):
        i = 0
        while not self.image_available and i<1000:
            time.sleep(0.001)
            i+=1

        if i<1000:
            self.image_available = False
            return self.image.copy()
        else:
            self.server_crash_detected = True
            return np.zeros_like(self.image)

    def get_image_list(self):
        self.recording = False
        return self.image_array, (self.IM_HEIGHT, self.IM_WIDTH)

    def is_server_crashed(self):
        return self.server_crash_detected

    def get_semantic_image(self):
        return self.semantic_image.copy()

    def get_road_highlight(self):
        sem_img = self.get_semantic_image()
        sem_img[sem_img == 6] = 255
        sem_img[sem_img == 7] = 255
        sem_img[sem_img < 255] = 0
        return sem_img

    def get_collisions(self):
        return self.collisions

    def get_current_control(self):
        return self.current_applied_control

    def front_side_proximity_detector(self):
        image = self.get_road_highlight()
        check = image[(image.shape[0]//3)*2, image.shape[1]//3:(image.shape[1]//3)*2:] == 0
        if check.any():
            return True
        return False

    def left_side_proximity_detector(self):
        image = self.get_road_highlight()
        check = image[-1, :(image.shape[1]//3):] == 0
        if check.any():
            return True
        return False

    def right_side_proximity_detector(self):
        image = self.get_road_highlight() 
        check = image[-1, (image.shape[1]//3)*2::] == 0
        if check.any():
            return True
        return False

    def has_crashed(self):
        if self.collisions:# physical collision
            return True

        image = self.get_road_highlight() # using semantic camera for out of road collision
        check = image[-1, ::] == 0 # image bottom line == 0 equals car out of road
        if check.all():
            return True

        return False

    def get_position(self):
        return self.vehicle.get_location()

    def get_speed(self):    # m/s
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def reset_position(self):
        self.action_by_id(0)
        while self.get_speed() >= 0.1:
            time.sleep(0.1)
        self.vehicle.set_location(self.start_position)
        self.collisions = []
        self.image_array = []
        self.recording = True
        time.sleep(0.1)

    def destroy(self):
        self.stop = True
        self.camera.stop()
        self.s_camera.stop()
        self.collision.stop()
        if not self.is_server_crashed():
            for actor in self.actor_list:
                actor.destroy()
