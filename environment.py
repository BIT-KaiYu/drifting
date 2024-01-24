from __future__ import print_function	# 即使在python2.X，使用print就得像python3.X那样加括号使用
import time
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')
import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from carla_tools import *
from tools import *
import argparse
from collections import deque
import pandas as pd
from scipy import signal

step_T_bound = (0.6,1)		# Boundary of throttle values 油门
step_S_bound = (-0.8,0.8)	# Boundary of the steering angle values 转向角

b1, a1 = signal.butter(3, 0.04, 'lowpass')   # 配置方向盘滤波
b2, a2 = signal.butter(3, 0.04, 'lowpass')   # 配置油门滤波

def draw_waypoints(world, route):
	x0 = route[0,0]
	y0 = route[0,1]
	for k in range(1,route.shape[0]):
		r = route[k,:]
		x1 = r[0]
		y1 = r[1]
		dx = x1-x0
		dy = y1-y0
		if math.sqrt(dx*dx+dy*dy) > 30:  # original 2.5
			x0 = x1
			y0 = y1
			begin = carla.Location(x = x1,y = y1, z = 0.2)
			angle = math.radians(r[2])
			end = begin + carla.Location(x=6*math.cos(angle), y=6*math.sin(angle))
			world.debug.draw_arrow(begin, end, arrow_size=12,life_time=90, color=carla.Color(238,18, 137,0))

class environment():
	def __init__(self, throttleSize=4, steerSize=9, traj_num = 0, collectFlag = False, model='dqn', vehicleNum=1):
		
		log_level = logging.INFO	# 日志系统
		
		logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

		logging.info('listening to server %s:%s', '127.0.0.1', 2000)
		
		self.refreshRoute(traj_num, vehicleNum)  # a series of caral.transform 读参考轨迹
		self.vehicleNum = vehicleNum

		if not collectFlag:
			start_location = carla.Location(x = self.route[0,0], y = self.route[0,1], z = 0.1)	 # 初始位姿
			start_rotation = carla.Rotation(pitch = 0, yaw = self.route[0,2], roll = 0)
		else:
			start_location = carla.Location()
			start_rotation = carla.Rotation()
		
		self.start_point = carla.Transform(location = start_location, rotation = start_rotation)  # type : Transform (location, rotation)
		
		self.client = carla.Client('127.0.0.1', 2000)	# 设置客户端IP和端口
		self.client.set_timeout(4.0)	# 设置在阻止网络调用并引发超时错误之前允许网络调用的最大时间。
		self.display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)	# 设置显示
		self.hud = HUD(1280, 720)	# 平视显示系统，是指以驾驶员为中心、盲操作、多功能仪表盘。
		self.world = World(self.client.get_world(), self.hud, 'vehicle.*', self.start_point, vehicleNum)
		self.clock = pygame.time.Clock()	# 计时
		self.minDis = 0	# 距离当前位置最近的路点的距离，匹配点的位置
		self.collectFlag = collectFlag
		self.traj_drawn_list = []

		self.control = carla.VehicleControl(
							throttle = 1,
							steer = 0.0,
							brake = 0.0,
							hand_brake = False,
							reverse = False,
							manual_gear_shift = False,
							gear = 0)	# 控车
		
		self.destinationFlag = False
		self.away = False
		self.collisionFlag = False
		self.waypoints_ahead = []	# 未走的路点
		self.waypoints_neighbor = []	# 周围的路点
		self.steer_history = deque(maxlen=20)
		self.throttle_history = deque(maxlen=20)
		self.slip_history = deque(maxlen=20)
		self.velocity_local = []	# 车体坐标系下的速度，滑移角
		self.model = model

		if model == 'dqn':
			self.step_T_pool = [step_T_bound[0]]
			self.step_S_pool = [step_S_bound[0]]
			t_step_rate = (step_T_bound[1]- step_T_bound[0])/throttleSize
			s_step_rate = (step_S_bound[1]- step_S_bound[0])/steerSize
			for i in range(throttleSize):
				self.step_T_pool.append(self.step_T_pool[-1]+t_step_rate)
			for i in range(steerSize):
				self.step_S_pool.append(self.step_S_pool[-1]+s_step_rate)	# 离散化
			print(self.step_T_pool)
			print(self.step_S_pool)
			self.tStateNum = len(self.step_T_pool)
			self.sStateNum = len(self.step_S_pool)

		self.e_heading = 0	# 航向偏差
		self.e_d_heading = 0
		self.e_dis = 0	# y方向的位置偏差
		self.e_d_dis = 0
		self.e_slip = 0	# 滑移角偏差】
		self.e_d_slip = 0
		self.e_vx = 0	# 横向偏差
		self.e_d_vx = 0
		self.e_vy = 0	# 纵向偏差
		self.e_d_vy = 0
		self.theta = 0  # 航向误差

		self.tg = 0	# 上一时刻的时间
		self.clock_history = 0 # pop the current location into self.waypoints_history every 0.2s

		self.k_heading = 0.1

		self.waypoints_ahead_local = []
		self.waypoints_history = deque(maxlen=5)	# 创建一个固定大小的队列。当新的元素加入并且这个队列已满的时候，最老的元素会自动被移除掉。
		self.waypoints_history_local = []

		self.last_steer = 0.0
		self.last_throttle = 0.0
		self.last_cur = 0.0
		self.road_type = 0

		self.tire_friction_array = np.arange(3,4.1,0.1) # [3,4], 11D 轮胎摩擦
		self.mass_array = np.arange(1700,1910,50) # array([1700, 1750, 1800, 1850, 1900]) 质量

		self.ori_physics_control = self.world.player.get_physics_control()	# 上一时刻的carla.VehiclePhysicsControl
		self.wheel_fl = self.ori_physics_control.wheels[0]
		self.wheel_fr = self.ori_physics_control.wheels[1]
		self.wheel_rl = self.ori_physics_control.wheels[2]
		self.wheel_rr = self.ori_physics_control.wheels[3]

		self.world.world.set_weather(carla.WeatherParameters.ClearNoon)

	def refreshRoute(self, traj_num, vehicleNum):
		# if vehicleNum == 1:
		traj = pd.read_csv('ref_trajectory/traj_' + str(traj_num) + '.csv')
		# else:
		# 	traj = pd.read_csv('ref_trajectory/traj_different_vehicles/' + str(vehicleNum) + '.csv')	# 读参考轨迹
		self.route = traj.values	# 取值
		self.route_x = self.route[:,0]	 # x
		self.route_y = self.route[:,1]	 # y
		self.route_length = np.zeros(self.route.shape[0])	# 曲线长度
		for i in range(1, self.route.shape[0]):
			dx = self.route_x[i-1] - self.route_x[i]
			dy = self.route_y[i-1] - self.route_y[i]
			self.route_length[i] = self.route_length[i-1] + np.sqrt(dx * dx + dy * dy)
		


	def step(self, actionID = 4, steer = 0.0, throttle=0.0, time_cost=0.5, manual_control = False, randomFlag = False):
		# apply the computed control commands, update endFlag and return state/reward
		if not manual_control:
			if self.model == 'dqn':
				self.control = self.getAction(actionID=actionID)	# 计算控制指令，油门和刹车
			else:
				self.control = self.getAction(steer=steer,throttle=throttle)

			if self.model == 'sac':
				self.control.steer = 0.1*self.control.steer + 0.9*self.last_steer
				self.control.throttle = 0.3*self.control.throttle + 0.7*self.last_throttle
			if self.model == 'ddpg':
				self.control.steer = 0.6*self.control.steer + 0.4*self.last_steer
				self.control.throttle = 0.3*self.control.throttle + 0.7*self.last_throttle
			# if self.model == 'ppo':
			# 	if np.array(self.steer_history).size > 2:
			# 		self.control.steer = 0.1 * self.control.steer + 0.9 * self.last_steer
			# 		self.control.throttle = 0.3 * self.control.throttle + 0.7 * self.last_throttle

			self.last_steer = self.control.steer
			self.last_throttle = self.control.throttle

			# 2022-5-4 滤波
			if np.array(self.steer_history).size > 12:
				self.steer_history.append(self.control.steer)	# 队列加上滤波前
				self.throttle_history.append(self.control.throttle)
				steer_history_array = signal.filtfilt(b1, a1, np.array(self.steer_history))	# array 滤波
				throttle_history_array = signal.filtfilt(b2, a2, np.array(self.throttle_history))
				steer_history_array = np.clip(steer_history_array, step_S_bound[0], step_S_bound[1])	# 限幅
				throttle_history_array = np.clip(throttle_history_array, step_T_bound[0], step_T_bound[1])
				self.control.steer = steer_history_array[-1]	# 从 array 中取出滤波后的控制量
				self.control.throttle = throttle_history_array[-1]
				self.steer_history.pop()	# 队列删掉滤波前
				self.throttle_history.pop()

			# self.control.steer = 0.0
			# self.control.throttle = 0.6

			self.world.player.apply_control(self.control)
			self.steer_history.append(self.control.steer)
			self.throttle_history.append(self.control.throttle)
			time.sleep(0.05)

		
		if manual_control and not self.collectFlag:
			control = self.world.player.get_control()
			self.steer_history.append(control.steer)
			self.throttle_history.append(control.throttle)
			time.sleep(0.05)
		
		newState = self.getState(randomFlag)

		if not self.collectFlag :
			self.collisionFlag = self.collisionDetect()

			slip = abs(self.velocity_local[2])

			reward = self.getReward(newState, time_cost, self.steer_history, self.throttle_history)

			return newState, slip, reward, self.collisionFlag, self.destinationFlag, self.away, self.control

		else:
			control = self.world.player.get_control()
			return newState, control
		
		
	def reset(self, traj_num = 0, collect_x = 0, collect_y = 0, collect_yaw = 0, randomPosition = False, testFlag = False, 
				test_friction = 3.5, test_mass = 1800.0, differentFriction=False, differentVehicles=False):
		# random change the tire friction and vehicle mass:
		if not testFlag:
			alpha_noise = 0.2
			self.tire_friction = random.uniform(test_friction*(1-alpha_noise), test_friction*(1+alpha_noise))
			self.mass = random.uniform(test_mass*(1-alpha_noise), test_mass*(1+alpha_noise))
		else:
			self.tire_friction = test_friction
			self.mass = test_mass
		
		if not differentFriction:
			self.wheel_fl.tire_friction = self.tire_friction
			self.wheel_fr.tire_friction = self.tire_friction
			self.wheel_rl.tire_friction = self.tire_friction
			self.wheel_rr.tire_friction = self.tire_friction
		else:
			self.wheel_fl.tire_friction = 2.8
			self.wheel_fr.tire_friction = 2.8
			self.wheel_rl.tire_friction = 4.2
			self.wheel_rr.tire_friction = 4.2

		wheels = [self.wheel_fl, self.wheel_fr, self.wheel_rl, self.wheel_rr]

		self.ori_physics_control.wheels = wheels
		if not differentVehicles:
			self.ori_physics_control.mass = float(self.mass)
		
		bbox = self.world.player.bounding_box.extent	# 通过获取车辆的包围盒获取车的尺寸
		print('vehicleNum', self.vehicleNum)
		print('size: ', bbox * 2)
		alpha_noise = 0.05
		if not testFlag:
			self.ori_physics_control.center_of_mass = carla.Vector3D(60.0, 0.0, -25.0) + carla.Vector3D(random.uniform(-1*alpha_noise*bbox.x*2, alpha_noise*bbox.x*2),0,random.uniform(-1*alpha_noise*bbox.z*2, alpha_noise*bbox.z*2))
		
		self.world.player.apply_physics_control(self.ori_physics_control)	# 设置轮胎摩擦和质量
		time.sleep(0.5)

		# detect:
		physics = self.world.player.get_physics_control()
		print('firction: {}, mass: {}'.format(physics.wheels[0].tire_friction, physics.mass))
		print('max_rpm: ', physics.max_rpm)
		print('center of mass: ', physics.center_of_mass.x, physics.center_of_mass.y, physics.center_of_mass.z)
		
		if not self.collectFlag:
			self.refreshRoute(traj_num, self.vehicleNum)
			if not randomPosition:
				start_location = carla.Location(x = self.route[0,0], y = self.route[0,1], z = 0.1)
				start_rotation = carla.Rotation(pitch = 0, yaw = self.route[0,2], roll = 0)
				velocity_local = [10,0]  # 5m/s
				angular_velocity = carla.Vector3D()
				
			else:
				k = np.random.randint(0,self.route.shape[0] - 100)
				start_location = carla.Location(x = self.route[k,0], y = self.route[k,1], z = 0.1)
				start_rotation = carla.Rotation(pitch = 0, yaw = self.route[k,2], roll = 0)
				velocity_local = [10, 0] 
				# angular_velocity = carla.Vector3D(z = self.route[k,6])
				angular_velocity = carla.Vector3D()
		else:
			start_location = carla.Location(x = collect_x, y=collect_y)
			start_rotation = carla.Rotation(yaw = collect_yaw)

		
		self.start_point = carla.Transform(location = start_location, rotation = start_rotation)  # type : Transform (location, rotation)
		ego_yaw = self.start_point.rotation.yaw

		# if not self.collectFlag:
		# 	if traj_num not in self.traj_drawn_list:
		# 		self.drawPoints()	# 画箭头
		# 		self.traj_drawn_list.append(traj_num)

		
		ego_yaw = ego_yaw/180.0 * 3.141592653
		transformed_world_velocity = self.velocity_local2world(velocity_local, ego_yaw)

		self.world.player.set_transform(self.start_point)	# RESET
		self.world.player.set_velocity(transformed_world_velocity)
		self.world.player.set_angular_velocity(angular_velocity)
		
		self.world.player.apply_control(carla.VehicleControl())

		self.world.collision_sensor.history = []
		self.away = False
		self.endFlag = False
		self.steer_history.clear()
		self.throttle_history.clear()
		self.slip_history.clear()
		self.waypoints_neighbor = []
		self.waypoints_ahead = []	# 未走的路点

		self.waypoints_ahead_local = [] # carla.location 10pts
		self.waypoints_history.clear()  # carla.location  5pts
		self.waypoints_history_local = []
		self.destinationFlag = False

		self.last_steer = 0.0
		self.last_throttle = 0.0
		self.last_cur = 0.0
		self.road_type = 0

		self.drived_distance = 0	# 已经走过的距离

		print('RESET!\n\n')
		
		return 0

	def getState(self, randomFlag = False):
		location = self.world.player.get_location()	# 获取carla.Actor（这里是车）的位置x,y,z
		angular_velocity = self.world.player.get_angular_velocity()	# 角速度
		transform = self.world.player.get_transform()	# 位姿
		ego_yaw = transform.rotation.yaw
		velocity_world = self.world.player.get_velocity()

		if randomFlag:
			location.x += random.gauss(0, 0.02)	# 均值和标准差
			location.y += random.gauss(0, 0.02)
			location.z += random.gauss(0, 0.02)
			ego_yaw += random.gauss(0, 0.2)
			velocity_world.x += random.gauss(0, 0.03)	# 均值和标准差
			velocity_world.y += random.gauss(0, 0.03)
			velocity_world.z += random.gauss(0, 0.03)

		if ego_yaw < 0:
			ego_yaw += 360
		if ego_yaw > 360:
			ego_yaw -= 360
		ego_yaw = ego_yaw/180.0 * 3.141592653	# 偏航角

		self.getNearby(location) # will update self.minDis 寻找匹配点
		self.getLocalHistoryWay(location, ego_yaw)	# 得到waypoints_history_local
		self.getLocalFutureWay(location, ego_yaw)	# 得到waypoints_ahead_local
		self.velocity_world2local(ego_yaw, velocity_world) # will update self.velocity_local 得到velocity_local，车体坐标系下的速度和滑移角

		ego_yaw = ego_yaw/3.141592653 * 180	# 度数
		if ego_yaw > 180:
			ego_yaw = -(360-ego_yaw)

		if self.collectFlag:
			state = [location.x, location.y, ego_yaw, self.velocity_local[0], self.velocity_local[1], self.velocity_local[2], angular_velocity.z]
			
			self.control = self.world.player.get_control()	# 获取控制量
			steer = self.control.steer
			ct = time.time()
			if ct - self.clock_history > 0.3:
				self.waypoints_history.append(np.array([location.x, location.y, steer, self.velocity_local[2]]))
				self.clock_history = ct	# 每0.3s采集一次数据

			return state
			
		else:
			dt = time.time() - self.tg
			self.e_d_dis = (self.minDis - self.e_dis) / dt
			self.e_dis = self.minDis	# 上一时刻的最小距离

			if self.e_dis > 15:	# rc-car 4
				self.away = True

			# error of heading:
			# 1. calculate the abs
			way_yaw = self.waypoints_ahead[0,2]
			# 2. update the way_yaw based on vector guidance field:
			vgf_left = self.vgf_direction(location)  
			# 3. if the vehicle is on the left of the nearst waypoint, according to the heading of the waypoint
			if vgf_left:
				way_yaw = math.atan(self.k_heading * self.e_dis)/3.141592653*180 + way_yaw
			else:
				way_yaw = -math.atan(self.k_heading * self.e_dis)/3.141592653*180 + way_yaw
			if way_yaw > 180:
				way_yaw = -(360-way_yaw)
			if way_yaw < -180:
				way_yaw += 360

			if ego_yaw*way_yaw > 0:
				e_heading = abs(ego_yaw - way_yaw)
			else:
				e_heading = abs(ego_yaw) + abs(way_yaw)
				if e_heading > 180:
					e_heading = 360 - e_heading
			# considering the +-:
			# waypoint to the vehicle, if clockwise, then +
			hflag = 1
			if ego_yaw*way_yaw > 0:
				if ego_yaw > 0:
					if abs(way_yaw) < abs(ego_yaw):
						hflag = -1
					else:
						hflag = 1
				if ego_yaw < 0:
					if abs(way_yaw) < abs(ego_yaw):
						hflag = 1
					else:
						hflag = -1
			else:
				if ego_yaw > 0:
					t_yaw = ego_yaw-180
					if way_yaw > t_yaw:
						hflag = -1
					else:
						hflag = 1
				else:
					t_yaw = ego_yaw + 180
					if way_yaw > t_yaw:
						hflag = -1
					else:
						hflag = 1
			e_heading = e_heading * hflag
			if e_heading * self.e_heading > 0:
				if e_heading > 0:
					self.e_d_heading = (e_heading - self.e_heading)/dt
				else:
					self.e_d_heading = -(e_heading - self.e_heading)/dt
			else:
				self.e_d_heading = (abs(e_heading) - abs(self.e_heading)) / dt
				
			self.e_heading = e_heading
			
			e_slip = self.velocity_local[2] - self.waypoints_ahead[0,5]
			self.e_d_slip = (e_slip - self.e_slip)/dt
			self.e_slip = e_slip

			e_vx = self.velocity_local[0] - self.waypoints_ahead[0,3]
			self.e_d_vx = (e_vx - self.e_vx)/dt
			self.e_vx = e_vx

			e_vy = self.velocity_local[1] - self.waypoints_ahead[0,4]
			self.e_d_vy = (e_vy - self.e_vy)/dt
			self.e_vy = e_vy

			self.control = self.world.player.get_control()

			steer = self.control.steer
			throttle = self.control.throttle
			
			ct = time.time()
			if ct - self.clock_history > 0.2:
				self.waypoints_history.append(np.array([location.x, location.y, steer, self.velocity_local[2]]))
				self.clock_history = ct

			vx = self.velocity_local[0]
			vy = self.velocity_local[1]
			e_d_slip = self.e_d_slip
			if math.sqrt(vx*vx + vy*vy) < 2: # if the speed is too small we ignore the error of slip angle
				e_slip = 0
				e_d_slip = 0

			dx = self.waypoints_ahead_local[1][0] - self.waypoints_ahead_local[0][0]
			dy = self.waypoints_ahead_local[1][1] - self.waypoints_ahead_local[0][1]
			if dx == 0:
				if dy < 0:
					self.theta = 2 * math.atan(1)
				else:
					self.theta = 6 * math.atan(1)
			else:
				self.theta = math.atan(dy / dx)
				if dx > 0:
					self.theta = -1 * self.theta
				if dx < 0:
					self.theta = 4 * math.atan(1) - self.theta
			self.theta = self.theta/math.atan(1)/4*180
			slip = self.velocity_local[2]
			self.slip_history.append(slip)

			# 2022-3-1修改
			# state = [steer, throttle , self.e_dis, self.e_d_dis, self.e_heading, self.e_d_heading, e_slip, e_d_slip,
			#         self.e_vx, self.e_d_vx, self.e_vy, self.e_d_vy]
			state = [self.e_dis, self.theta, vx, vy, slip]	# 2022-5-4 删除控制量
			state.extend([k[0] for k in self.waypoints_ahead_local]) #x
			state.extend([k[1] for k in self.waypoints_ahead_local]) #y
			# state.extend([k[2] for k in self.waypoints_ahead_local]) #slip
			self.tg = time.time()

			return state

	# 2022-3-1修改
	def getReward(self, state, time_cost, steer_history, throttle_history):
		e_dis = state[0]
		theta = state[1]
		vx = self.velocity_local[0]
		vy = self.velocity_local[1]
		v = math.sqrt(vx*vx + vy*vy)
		slip = abs(self.velocity_local[2])
		cur = PJcurvature(self.waypoints_ahead[:,0],self.waypoints_ahead[:,1])
		self.last_cur = cur

		r_collision = -50 if (self.collisionFlag) else 0
		r_dis = np.exp(-0.5 * e_dis)
		r_theta = 0.5 * np.exp(-0.05 * abs(theta))
		r_v = 2 * (np.exp(-0.04 * abs(v - 30)) if (v < 30) else 1)
		if (abs(cur) > 0.04):
			# r_slip = 20 * (np.exp(-0.05 * abs(slip - 20)) if (slip < 20) else 1)
			r_slip = 0.4 * slip
			self.road_type = 2
		elif (abs(cur) < 0.01):
			r_slip = -0.1 * slip
			self.road_type = 0
		else:
			r_slip = 0
			self.road_type = 1
		
		reward = r_collision + r_dis + r_theta + r_v + r_slip

		print("e_dis: ", e_dis)
		print("theta: ", theta)
		print("v: ", v)
		print("r_collision: ", r_collision)
		print("r_dis: ", r_dis)
		print("r_theta: ", r_theta)
		print("r_v: ", r_v)
		print("r_slip: ", r_slip, "  cur: ", cur)

		return reward

	def getNearby(self, egoLocation):

		self.waypoints_ahead = [] 
		self.waypoints_neighbor = []
		# egoLocation = self.world.player.get_location()
		dx_array = self.route_x - egoLocation.x
		dy_array = self.route_y - egoLocation.y
		dis_array = np.sqrt(dx_array * dx_array + dy_array * dy_array)
		self.minDis = np.amin(dis_array)	# 寻找路点上距离当前位置最近的点，匹配点
		_ = np.where(dis_array == self.minDis)
		index = _[0][0]  # index for the min distance to all waypoints.

		x1 = carla.Location(self.route_x[index], self.route_y[index], 0)
		x22 = carla.Location(self.route_x[index + 1], self.route_y[index + 1], 0)
		if index == 0:
			x21 = x1
		else:
			x21 = carla.Location(self.route_x[index - 1], self.route_y[index - 1], 0)
		self.minDis = min(point2line(egoLocation, x1, x21), point2line(egoLocation, x1, x22)) 	# 点到直线距离

		self.drived_distance = self.route_length[index]	# 已经走过的距离
		self.waypoints_ahead = self.route[index:,:]	# 未走的路点

		if index >= 20:
			index_st = index - 20
		else:
			index_st = 0
		self.waypoints_neighbor = self.route[index_st:,:]
		self.traj_index = index	# 匹配点的序号


	def drawPoints(self):
		draw_waypoints(self.world.player.get_world(), self.route)


	def render(self):
		# show ROS client window by pygame
		self.world.tick(self.clock, self.e_dis, self.theta, self.velocity_local[2], self.last_cur)
		self.world.render(self.display)
		pygame.display.flip()


	def velocity_world2local(self,yaw,velocity_world):
		# velocity_world = self.world.player.get_velocity()
		vx = velocity_world.x
		vy = velocity_world.y
		yaw = -yaw
		
		local_x = float(vx * math.cos(yaw) - vy * math.sin(yaw))
		local_y = float(vy * math.cos(yaw) + vx * math.sin(yaw))	# 车体坐标系下的速度
		if local_x != 0:
			slip_angle = math.atan(local_y/local_x)/3.1415926*180	# 滑移角
		else:
			slip_angle = 0
		
		self.velocity_local = [local_x,local_y,slip_angle]

	def velocity_local2world(self, velocity_local, yaw):
		vx = velocity_local[0]
		vy = velocity_local[1]

		world_x = vx * math.cos(yaw) - vy * math.sin(yaw)
		world_y = vy * math.cos(yaw) + vx * math.sin(yaw)

		return carla.Vector3D(world_x,world_y,0)

	def collisionDetect(self):
		if self.world.collision_sensor.history:
			return True
		else:
			return False

	def getAction(self,actionID=4,steer=0.0, throttle=0.0):
		if self.model == 'dqn':
			throttleID = int(actionID / self.sStateNum)
			steerID = int(actionID % self.sStateNum)
				
			self.control = carla.VehicleControl(
								throttle = self.step_T_pool[throttleID],
								steer = self.step_S_pool[steerID],
								brake = 0.0,
								hand_brake = False,
								reverse = False,
								manual_gear_shift = False,
								gear = 0)
		else:
			self.control = carla.VehicleControl(
							throttle = throttle,
							steer = steer,
							brake = 0.0,
							hand_brake = False,
							reverse = False,
							manual_gear_shift = False,
							gear = 0)
		return self.control

	def coordinateTransform(self,egoLocation,yaw):
		# transfer the nearest waypoint to the local coordinate.
		way_x = self.waypoints_ahead[0,0]
		way_y = self.waypoints_ahead[0,1]
		yaw = -yaw

		dx = way_x - egoLocation.x
		dy = way_y - egoLocation.y

		nx = dx * math.cos(yaw) - dy * math.sin(yaw)
		ny = dy * math.cos(yaw) + dx * math.sin(yaw)

		if nx > 0 and ny > 0:
			return 1
		elif nx> 0 and ny < 0:
			return 2
		elif nx<0 and ny < 0:
			return 3
		elif nx<0 and ny>0:
			return 4

	def getLocalFutureWay(self,egoLocation,yaw):
		# transfer the future waypoints (#10) to the local coordinate.
		# x, y, slip (degree)

		ways = self.waypoints_ahead[0:-1:5,:]  # filter to 1m between way pts 
		if ways.shape[0] < 11:
			self.destinationFlag = True
			self.waypoints_ahead_local = []
			yaw = -yaw
		
			for w in ways[0:ways.shape[0]]: 
				
				wx = w[0]
				wy = w[1]
				w_slip = w[5]
				dx = wx - egoLocation.x
				dy = wy - egoLocation.y

				nx = dx * math.cos(yaw) - dy * math.sin(yaw)
				ny = dy * math.cos(yaw) + dx * math.sin(yaw)
				self.waypoints_ahead_local.append(np.array([nx,ny,w_slip]))
			for w in ways[0:10-ways.shape[0]]: 
				self.waypoints_ahead_local.append(np.array([nx,ny,w_slip]))
		else:
			self.waypoints_ahead_local = []
			yaw = -yaw
		
			for w in ways[0:10]: 
				
				wx = w[0]
				wy = w[1]
				w_slip = w[5]
				dx = wx - egoLocation.x
				dy = wy - egoLocation.y

				nx = dx * math.cos(yaw) - dy * math.sin(yaw)
				ny = dy * math.cos(yaw) + dx * math.sin(yaw)
				self.waypoints_ahead_local.append(np.array([nx,ny,w_slip]))


	def getLocalHistoryWay(self,egoLocation,yaw):
		# x, y, steer, slip (degree)
		ways = self.waypoints_history
		yaw = -yaw
		self.waypoints_history_local = []
		if len(ways) < 5:
			for i in range(5 - len(ways)):
				self.waypoints_history_local.append(np.array([0,0,0,0]))
		
		for w in ways:
			wx = w[0]
			wy = w[1]
			w_steer = w[2]
			w_slip = w[3]
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y

			nx = dx * math.cos(yaw) - dy * math.sin(yaw)
			ny = dy * math.cos(yaw) + dx * math.sin(yaw)
			self.waypoints_history_local.append(np.array([nx,ny,w_steer,w_slip]))

	def vgf_direction(self,egoLocation):
		way_x = self.waypoints_ahead[0,0]
		way_y = self.waypoints_ahead[0,1]
		yaw = -self.waypoints_ahead[0,2]/180.0 * 3.141592653
		
		dx = egoLocation.x - way_x
		dy = egoLocation.y - way_y

		nx = dx * math.cos(yaw) - dy * math.sin(yaw)
		ny = dy * math.cos(yaw) + dx * math.sin(yaw)

		if ny < 0:
			return True
		else:
			return False


	

