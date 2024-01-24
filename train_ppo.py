from pickle import FALSE
from re import T
import sys
from environment import *
from PPO_GAEAgent import *
import time
import random
import pygame
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from agents.navigation.basic_agent import BasicAgent
from tools import getHeading, bool2num

######## PPO #######
if __name__ == "__main__":

	# 1、设置参数
	# argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
	# argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
	parser = argparse.ArgumentParser()	# 创建一个解析对象

	parser.add_argument('--has_continuous_action_space', default=True, type=bool) # continuous action space; else discrete
	parser.add_argument('--action_std', default=0.5, type=float)	# starting std for action distribution (Multivariate Normal)
	parser.add_argument('--action_std_decay_rate', default=0.01, type=float)	# linearly decay action_std (action_std = action_std - action_std_decay_rate)
	parser.add_argument('--min_action_std', default=0.1, type=float)	# minimum action_std (stop decay after action_std <= min_action_std)

	parser.add_argument('--iteration', default=10000, type=int)  # num of  games
	parser.add_argument('--action_std_decay_freq', default=30, type=int)	# action_std decay frequency
	parser.add_argument('--save_model_freq', default=10, type=int)	# save model frequency
	parser.add_argument('--update_timestep', default=2048, type=int)	# update policy every n timesteps，buffer大小
	parser.add_argument('--K_epochs', default=10, type=int)  # update policy for K epochs in one PPO update
	parser.add_argument('--batch_size', default=64, type=int)

	parser.add_argument('--eps_clip', default=0.2, type=float) # clip parameter for PPO
	parser.add_argument('--gamma', default=0.99, type=float)  # discount factor
	parser.add_argument('--lamda', default=0.95, type=float)  # discount factor
	parser.add_argument('--lr_actor', default=0.0004, type=float)  # learning rate for actor network
	parser.add_argument('--lr_critic', default=0.001, type=float)  # learning rate for critic network

	parser.add_argument('--seed', default=1, type=int)	# set random seed if required (0 = no random seed)
	parser.add_argument('--load', default=False, type=bool)  # load model

	# 把parser中设置的所有"add_argument"给返回到args子类实例当中
	args = parser.parse_args()	# 进行解析

	###################### logging ######################
	log_dir = "PPO_logs"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	run_num = 0
	current_num_files = next(os.walk(log_dir))[2]
	run_num = len(current_num_files)

	log_f_name = log_dir + "/PPO_log_" + str(run_num) + ".csv"

	print("current logging run number is : ", run_num)
	print("logging at : " + log_f_name)

	# logging file
	log_f = open(log_f_name, "w+")
	log_f.write('episode,run_num_trained,time_cost,count,destinationFlag,avg_speed,avg_cross_track_error,avg_theta,avg_slip_angle_corner,slip_angle_straight,avg_throttle,avg_reward\n')

	################### checkpointing ###################
	checkpoint_path = "PPO_weights" + "/PPO.pth"

	# 2、环境初始化
	print(1)
	pygame.init()	# 检查电脑上一些需要的硬件调用接口、基础功能是否有问题
	print(2)
	pygame.font.init()	# 该函数用于初始化字体模块
	print(3)
	env = environment(traj_num=6, model='ppo', vehicleNum=1)	# carla环境初始化，路线选择0-6; 车辆选择；2022-3-3修改

	# 3、获取当前状态
	action_dim = 2	# 动作维数
	state = env.getState()
	state_dim = len(state)	# 状态维数
	print('action_dimension:', action_dim, ' & state_dimension:', state_dim)

	destinationFlag = False
	collisionFlag = False
	awayFlag = False
	carla_startFlag = False

	# 4、定义agent
	agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.lamda, args.K_epochs, args.batch_size, args.eps_clip, args.has_continuous_action_space, args.action_std)
	# 5、加载网络参数，默认不加载
	if args.load:
		print("load checkpoint path : " + checkpoint_path)
		agent.load(checkpoint_path)

	print("--------------------------------------------------------------------------------------------")

	# states_min = np.ones((1,25)) * 100
	# states_max = np.ones((1,25)) * -100
	states_min = np.array([[0,-90,0,-10,-30,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50]])
	states_max = np.array([[15,90,40,10,30,80,80,80,80,80,80,80,80,80,80,50,50,50,50,50,50,50,50,50,50]])

	time_step = 0	# 总步数，用于确定是否更新网络参数
	run_num_trained = 0	# 训练次数
	run_time_0 = time.time()
	# 6、训练
	for i in range(args.iteration):	# 对于每一轮游戏
		state = env.reset(traj_num=6, randomPosition=False, testFlag=False)	# 1.重置环境，获取当前状态
		t0 = time.time()	# 获取当前时刻
		first_step_pass = False	# 第一步的flag

		count = 0	# 步数
		time_cost = 0  # 所需时间
		cte = 0	# 平均位置误差
		theta = 0
		speed = 0	# 平均速度
		slip = 0
		slip1 = 0	# 直道
		slip2 = 0	# 大弯
		avg_throttle = 0 
		count1 = 0
		count2 = 0
		ep_r = 0	# 总奖励
		avg_reward = 0	# 平均奖励

		while(True):
			env.render()	# show ROS client window by pygame，显示窗口

			# start training when the carla env is ready, before that we loop:
			tmp_control = env.world.player.get_control()
			if tmp_control.throttle == 0 and carla_startFlag==False:
				tmp_control = carla.VehicleControl(
								throttle = 0.5,
								steer = 0,
								brake = 0.0,
								hand_brake = False,
								reverse = False,
								manual_gear_shift = False,
								gear = 0)
				env.world.player.apply_control(tmp_control)
				continue
			carla_startFlag = True	# 判断carla环境已经OK

			if time.time()-t0 < 0.5:	# 游戏开始0.5s内
				env.world.collision_sensor.history = []	# 碰撞传感器清空
			if time.time()-t0 > 0.5:	# 0.5s后

				if not first_step_pass:	# 2.选择动作
					steer = 0.0
					throttle = 0.0
					hand_brake = False	# 第一次
				else:
					action = agent.select_action(tState) # 选择动作
					action = np.reshape(action, [1, 2])	# 塑形

					steer = action[0, 0]
					throttle = action[0, 1]
					print("mapped steer: ", steer, ", throttle: ", throttle)

				time_cost = time.time() - t0
				next_state, slip_angle, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer, throttle=throttle, time_cost=time_cost, randomFlag = True)	# 3.执行动作
				print("actual steer: ", control.steer, ", throttle: ", control.throttle)
				print("-------------------------------------------------------------------------------")

				next_state = np.reshape(np.array(next_state), [1, state_dim])	# 下一时刻的状态

				count += 1  # 第几步动作
				time_step += 1	# 总步数
				endFlag = collisionFlag or destinationFlag or awayFlag	# 是否结束：碰撞，到达终点，远离路线

				if first_step_pass:
					agent.buffer.rewards.append(reward)  # 存buffer
					agent.buffer.is_terminals.append(endFlag)

				cte += abs(next_state[0,0])
				theta += abs(next_state[0,1])	# 累计角度偏差
				vx = env.velocity_local[0]	# 记录数据,速度m/s
				vy = env.velocity_local[1]
				speed += np.sqrt(vx*vx + vy*vy)
				slip += abs(slip_angle)
				if env.road_type == 0:
					slip1 += abs(slip_angle)
					count1 += 1
				elif env.road_type == 2:
					slip2 += abs(slip_angle)
					count2 += 1
				avg_throttle += control.throttle
				ep_r += reward	# 累计奖励

				tState = next_state
				# 状态归一化
				# agent.ob_rms.update(next_state)
				# tState = (next_state - agent.ob_rms.mean) / np.sqrt(agent.ob_rms.var + 1e-8)	# 滚动求均值方差
				
				# for i in range(state_dim):
				# 	if(next_state[0][i] > states_max[0][i]):
				# 		states_max[0][i] = next_state[0][i]
				# 	if(next_state[0][i] < states_min[0][i]):
				# 		states_min[0][i] = next_state[0][i]
				# print(states_max)
				# print(states_min)	# 测试最值

				tState = np.clip(tState, states_min, states_max)
				for j in range(state_dim):
					tState[0][j] = (tState[0][j] - states_min[0][j])/(states_max[0][j] - states_min[0][j])

				# update PPO agent
				if time_step % args.update_timestep == 0:
					print("------------------------------------------------------------------------")
					print("TRAINING, the time_step is: %d" % time_step)
					agent.update()  # 走了很多步了，更新一下参数
					run_num_trained += 1
					print("------------------------------------------------------------------------")

				# if continuous action space; then decay action std of ouput action distribution
				if args.has_continuous_action_space and time_step % (args.action_std_decay_freq * args.update_timestep) == 0:
					agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)  # 标准差衰减一下

				# save model weights
				if time_step % (args.save_model_freq * args.update_timestep) == 0:
					print("------------------------------------------------------------------------")
					model_dir = "PPO_model_0"
					if not os.path.exists(model_dir):
						os.makedirs(model_dir)
					savepoint_path = model_dir + "/PPO_{}.pth".format(run_num_trained)
					print("saving model at : " + savepoint_path)
					agent.save(savepoint_path)
					print("model saved")
					print("------------------------------------------------------------------------")

				if endFlag:
					break	# 游戏结束

				first_step_pass = True	# 第一步执行完了

		run_time = time.time() - run_time_0
		time_cost = time.time() - t0	# 一轮游戏所需时间，后续还需要对这轮游戏进行处理	
		cte = cte/count
		theta = theta/count
		speed = speed / count	# 平均速度
		slip = slip/count
		slip1 = slip1/count1 if (count1 != 0) else 0
		slip2 = slip2/count2 if (count2 != 0) else 0
		avg_reward = ep_r/count

		log_f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i+1, run_num_trained, time_cost, count, bool2num(destinationFlag), speed, cte, theta, slip2, slip1, avg_throttle/count, avg_reward))
		log_f.flush()
		
		print("Ep_i: %d, time_step: %d,  the ep_r is: %.2f" % (i, time_step, ep_r))
		print("------------------------------------------------------------------------")

	log_f.close()