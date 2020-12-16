#!/usr/bin/env python3.7.7
# Note: use Python 3.7.7

import os
import sys
import glob
import random

# unused imports
import cv2 as cv
import numpy as np
import time

from utils.config import load_data

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
	sys.path.append(	glob.glob('./PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
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


CONFIGFILE = "config.json"


class Car:

	def process_img(self, raw_image):
		i = np.array(raw_image.raw_data)
		i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4)) #RGBA
		self.image = cv.cvtColor(i2, cv.COLOR_RGBA2RGB)
		return

	def process_img_semantic(self, data):
		i = np.array(data.raw_data)
		i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4)) #RGBA
		self.semantic_image = i2[:,:,2]
		return

	def __init__(self):
		self.IM_WIDTH = 640
		self.IM_HEIGHT = 480
		self.actor_list = []

		settings = load_data(CONFIGFILE)

		initialized = False
		while not initialized:
			try:
				connection_data = settings["connection"]
				client = carla.Client(connection_data["server_ip"], connection_data["server_port"])
				client.set_timeout(10.0)

				world = client.get_world()
				blueprint_library = world.get_blueprint_library()

				bp = blueprint_library.filter("model3")[0]	#model3

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
				#print("camera initialized")

				# semantic camera initialization
				cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
				cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
				cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
				spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
				self.s_camera = world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
				self.actor_list.append(self.s_camera)
				self.s_camera.listen(lambda data: self.process_img_semantic(data))
				#print("semantic camera initialized")

				# collision initialization
				coll_bp = blueprint_library.find("sensor.other.collision")
				self.collision = world.spawn_actor(coll_bp, spawn_point, attach_to=self.vehicle)
				self.actor_list.append(self.collision)
				self.collisions = []
				self.collision.listen(lambda data: self.collisions.append(data))
				#print("collision sensor initialized")

			except RuntimeError:
				print("Init phase failed, check server connection")
				initialized = False
				time.sleep(30)

			initialized = True

		time.sleep(3)

	def _action_to_car_values(self, act):	# 0=stop, 1=forward, 2=left, 3=right
		if act == 0:
			return 0, 0, 1, False	#STOP
		elif act == 1:
			return 1, 0, 0, False	#FORWARD
		elif act == 2:
			return 0.5, -0.5, 0, False #LEFT
		elif act == 3:
			return 0.5, 0.5, 0, False #RIGHT
		else:
			raise ValueError('`action` should be between 0 and 3.')

	def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, reverse=False):	# throttle 0:1, steer -1:1, brake 0:1
		self.current_applied_control = (throttle, steer, brake, reverse)
		#self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=False, manual_gear_shift=True, gear=1))
		self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=False))

	def action_by_id(self, act):
		self.current_control = act
		throttle, steer, brake, reverse = self._action_to_car_values(act)
		self.apply_control(throttle, steer, brake, reverse)

	def get_im_sizes(self):
		return self.IM_WIDTH, self.IM_HEIGHT

	def get_image(self):
		return self.image.copy()

	def get_semantic_image(self):
		return self.semantic_image.copy()

	def get_road_highlight(self):
		sem_img = self.get_semantic_image()
		sem_img[ sem_img==6 ] = 255
		sem_img[ sem_img==7 ] = 255
		sem_img[ sem_img < 255 ] = 0
		return sem_img

	def get_collisions(self):
		return self.collisions

	def get_current_control(self):
		return self.current_applied_control

	def has_crashed(self):
		val = False

		#physical collision
		if self.collisions:
			val = True

		#camera bottom line black == out of road
		image = self.get_image()
		image[self.get_road_highlight() < 128] = 0
		check = image[-1,::] == 0
		if check.all():
			val = True
		
		return val

	def autopilot(self, state):
		carla.vehicle.set_autopilot(state)

	def destroy(self):
		self.camera.stop()
		self.s_camera.stop()
		self.collision.stop()
		for actor in self.actor_list:
			actor.destroy()