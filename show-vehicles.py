import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
 
import carla
from carla import ColorConverter as cc
 
actor_list =  []
try:
	#create client
	client = carla.Client('127.0.0.1', 2000)
	client.set_timeout(2.0)
	#world connection
	world = client.get_world() 
	#get blueprint libarary
	blueprint_library = world.get_blueprint_library()
      
	# 单一个车在图0
	location = carla.Location(x=-5, y=-10, z=5)
	rotation = carla.Rotation(pitch=0.0, yaw=90, roll=0.0)
	transform = carla.Transform(location,rotation)
	bp = blueprint_library.find("vehicle.audi.a2")
	print(bp)
	vehicle = world.spawn_actor(bp,transform)
	vehicle.set_autopilot(enabled=False)
	actor_list.append(vehicle)

	# location = carla.Location(x=0, y=-10.5, z=5)
	# rotation = carla.Rotation(pitch=0.0, yaw=-90, roll=0.0)
	# transform = carla.Transform(location,rotation)
	# bp = blueprint_library.find("vehicle.audi.a2")
	# print(bp)
	# vehicle = world.spawn_actor(bp,transform)
	# vehicle.set_autopilot(enabled=False)
	# actor_list.append(vehicle)

	# location = carla.Location(x=0, y=-13, z=5)
	# rotation = carla.Rotation(pitch=0.0, yaw=-30, roll=0.0)
	# transform = carla.Transform(location,rotation)
	# bp = blueprint_library.find("vehicle.citroen.c3")
	# print(bp)
	# vehicle = world.spawn_actor(bp,transform)
	# vehicle.set_autopilot(enabled=False)
	# actor_list.append(vehicle)

	# location = carla.Location(x=0, y=-16, z=5)
	# rotation = carla.Rotation(pitch=0.0, yaw=-30, roll=0.0)
	# transform = carla.Transform(location,rotation)
	# bp = blueprint_library.find("vehicle.chevrolet.impala")
	# print(bp)
	# vehicle = world.spawn_actor(bp,transform)
	# vehicle.set_autopilot(enabled=False)
	# actor_list.append(vehicle)

	# location = carla.Location(x=0, y=-19, z=5)
	# rotation = carla.Rotation(pitch=0.0, yaw=-30, roll=0.0)
	# transform = carla.Transform(location,rotation)
	# bp = blueprint_library.find("vehicle.volkswagen.t2")
	# print(bp)
	# vehicle = world.spawn_actor(bp,transform)
	# vehicle.set_autopilot(enabled=False)
	# actor_list.append(vehicle)

	# location = carla.Location(x=0, y=-22, z=5)
	# rotation = carla.Rotation(pitch=0.0, yaw=-30, roll=0.0)
	# transform = carla.Transform(location,rotation)
	# bp = blueprint_library.find("vehicle.carlamotors.carlacola")
	# print(bp)
	# vehicle = world.spawn_actor(bp,transform)
	# vehicle.set_autopilot(enabled=False)
	# actor_list.append(vehicle)
 
	time.sleep(600)
finally:
	client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
	print("All cleaned up!")
