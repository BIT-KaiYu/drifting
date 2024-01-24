import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.linalg as LA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
steer_range = (-0.8,0.8)
throttle_range = (0.6,1.0)

def getHeading(env):
	transform = env.world.player.get_transform()
	ego_yaw = transform.rotation.yaw
	if ego_yaw < 0:
		ego_yaw += 360
	if ego_yaw > 360:
		ego_yaw -= 360
	if ego_yaw > 180:
		ego_yaw = -(360-ego_yaw)
	return ego_yaw

def bool2num(flag):
	if flag == True:
		return 1
	else:
		return 0

def point2line(point0, point1, point2):
	if ((point1.x == point2.x) and (point1.y == point2.y)):
		dx = point0.x - point1.x
		dy = point0.y - point1.y
		return np.sqrt(dx * dx + dy * dy)
	else:
		return abs(((point2.x - point1.x)*(point0.y - point1.y) - (point0.x - point1.x)*(point2.y - point1.y)) / 
				np.sqrt((point2.x - point1.x) * (point2.x - point1.x) + (point2.y - point1.y) * (point2.y - point1.y)))

def PJcurvature(roadx,roady):
	x = [roadx[0],roadx[1],roadx[2]]
	y = [roady[0],roady[1],roady[2]]
	t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
	t_b = LA.norm([x[2]-x[1],y[2]-y[1]]) 
	M = np.array([
		[1, -t_a, t_a**2],
		[1, 0,    0     ],
		[1,  t_b, t_b**2]
	])
	a = np.matmul(LA.inv(M),x)
	b = np.matmul(LA.inv(M),y)
	kappa1 = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)

	x = [roadx[1],roadx[3],roadx[5]]
	y = [roady[1],roady[3],roady[5]]
	t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
	t_b = LA.norm([x[2]-x[1],y[2]-y[1]]) 
	M = np.array([
		[1, -t_a, t_a**2],
		[1, 0,    0     ],
		[1,  t_b, t_b**2]
	])
	a = np.matmul(LA.inv(M),x)
	b = np.matmul(LA.inv(M),y)
	kappa2 = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)

	x = [roadx[2],roadx[5],roadx[8]]
	y = [roady[2],roady[5],roady[8]]
	t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
	t_b = LA.norm([x[2]-x[1],y[2]-y[1]]) 
	M = np.array([
		[1, -t_a, t_a**2],
		[1, 0,    0     ],
		[1,  t_b, t_b**2]
	])
	a = np.matmul(LA.inv(M),x)
	b = np.matmul(LA.inv(M),y)
	kappa3 = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)

	print(kappa1)
	print(kappa2)
	print(kappa3)

	return (kappa1 + kappa2 + kappa3)/3
