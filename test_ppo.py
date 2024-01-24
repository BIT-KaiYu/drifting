from environment import *
from PPO_GAEAgent import *
import time
import pygame
from tools import getHeading, bool2num

if __name__ == "__main__":

    # 1、设置参数
    # argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
    # argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
    parser = argparse.ArgumentParser()  # 创建一个解析对象

    parser.add_argument('--has_continuous_action_space', default=True, type=bool)  # continuous action space; else discrete
    parser.add_argument('--action_std', default=0.0001, type=float)  # starting std for action distribution (Multivariate Normal)
    parser.add_argument('--K_epochs', default=10, type=int)  # update policy for K epochs in one PPO update
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eps_clip', default=0.2, type=float)  # clip parameter for PPO
    parser.add_argument('--gamma', default=0.99, type=float)  # discount factor
    parser.add_argument('--lamda', default=0.95, type=float)  # discount factor
    parser.add_argument('--lr_actor', default=0.0003, type=float)  # learning rate for actor network
    parser.add_argument('--lr_critic', default=0.001, type=float)  # learning rate for critic network

    parser.add_argument('--seed', default=1, type=int)  # set random seed if required (0 = no random seed)
    parser.add_argument('--load', default=True, type=bool)  # load model

    # 把parser中设置的所有"add_argument"给返回到args子类实例当中
    args = parser.parse_args()  # 进行解析

    np.random.seed(args.seed)

    ###################### logging ######################
    log_dir = "PPO_tests/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    log_f_name = log_dir + "PPO_log_" + str(run_num) + ".csv"

    print("current logging run number is : ", run_num)
    print("logging at : " + log_f_name)

    # logging file
    log_f = open(log_f_name, "w+")
    # 2022-3-1修改
    log_f.write('time,world_x,world_y,world_heading,local_vx,local_vy,total_v,slip_angle,cte,hte,'
                'traj_index,reward,steer,throttle,collisionFlag,desitinationFlag,awayFlag,abstheta,road_type,curvature,yaw_rate,roll_angle\n')

    directory = "PPO_weights"
    checkpoint_path = directory + "/PPO.pth"
    print("save checkpoint path : " + checkpoint_path)

    #####################################################

    # 2、环境初始化
    print(1)
    pygame.init()  # 检查电脑上一些需要的硬件调用接口、基础功能是否有问题
    print(2)
    pygame.font.init()  # 该函数用于初始化字体模块
    print(3)
    env = environment(traj_num=6, model='ppo')  # carla环境初始化，路线选择0-6; 2022-3-3修改

    # 3、获取当前状态
    action_dim = 2  # 动作维数
    state = env.getState()
    state_dim = len(state)  # 状态维数
    print('action_dimension:', action_dim, ' & state_dimension:', state_dim)

    destinationFlag = False
    collisionFlag = False
    awayFlag = False
    carla_startFlag = False

    # 4、定义agent
    agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.lamda, args.K_epochs, 
                args.batch_size, args.eps_clip, args.has_continuous_action_space, args.action_std)

    # 5、加载网络参数，默认不加载
    if args.load:
        agent.load(checkpoint_path)

    print("----------------------------------------------------------------------------")

    # 6、训练
    state = env.reset(traj_num=6, randomPosition=False, testFlag=True)  # 1.重置环境，获取当前状态
    t0 = time.time()  # 获取当前时刻
    first_step_pass = False  # 第一步的flag
    count = 0

    states_min = np.array([[0,-90,0,-10,-30,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50]])
    states_max = np.array([[15,90,40,10,30,80,80,80,80,80,80,80,80,80,80,50,50,50,50,50,50,50,50,50,50]])

    while (True):
        env.render()  # show ROS client window by pygame，显示窗口

        # start training when the carla env is ready, before that we loop:
        tmp_control = env.world.player.get_control()
        if tmp_control.throttle == 0 and carla_startFlag == False:
            tmp_control = carla.VehicleControl(
                throttle=0.5,
                steer=0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0)
            env.world.player.apply_control(tmp_control)
            continue
        carla_startFlag = True  # 判断carla环境已经OK

        if time.time() - t0 < 0.5:  # 游戏开始0.5s内
            env.world.collision_sensor.history = []  # 碰撞传感器清空
        if time.time() - t0 > 0.5:  # 0.5s后

            if not first_step_pass:  # 2.选择动作
                steer = 0.0
                throttle = 0.0
                hand_brake = False  # 第一次
            else:
                action = agent.select_action(tState)  # 选择动作
                action = np.reshape(action, [1, 2])  # 塑形

                steer = action[0, 0]
                throttle = action[0, 1]
                print("mapped steer: ", steer, ", throttle: ", throttle)

            time_cost = time.time() - t0
            next_state, slip_angle, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer, throttle=throttle, time_cost=time_cost, randomFlag = False)  # 3.执行动作
            next_state = np.reshape(next_state, [1, state_dim])  # 下一时刻的状态
            endFlag = collisionFlag or destinationFlag or awayFlag  # 是否结束：碰撞，到达终点，远离路线
            tState = next_state  # 4.获取下一时刻状态

            count += 1
            # pygame.image.save(env.display, "images/" + str(count).zfill(6) + '.png')

            # prepare the state information to be saved
            t = time.time() - t0
            location = env.world.player.get_location()
            wx = location.x
            wy = location.y
            course = getHeading(env)
            vx = env.velocity_local[0]
            vy = env.velocity_local[1]
            speed = np.sqrt(vx * vx + vy * vy) * 3.6
            slip_angle = env.velocity_local[2]
            cte = tState[0, 0]
            theta = tState[0,1]
            traj_index = env.traj_index
            steer = control.steer
            throttle = control.throttle
            cf = bool2num(collisionFlag)
            df = bool2num(destinationFlag)
            af = bool2num(awayFlag)
            print("time:  ", t)

            # 2023-5-4新增,用于描述车辆稳定性
            yaw_rate = env.world.player.get_angular_velocity().z
            roll_angle = env.world.player.get_transform().rotation.roll
            if roll_angle < -180:
                roll_angle += 360
            if roll_angle > 180:
                roll_angle -= 360

            first_step_pass = True  # 第一步执行完了

            log_f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format
                        (t,wx,wy,course,vx,vy,speed,slip_angle,cte,theta,traj_index,reward,steer,throttle,cf,df,af,abs(theta),env.road_type,env.last_cur,yaw_rate,roll_angle))
            log_f.flush()
            print("-------------------------------------------------------------------------------")

            if endFlag:
                break  # 游戏结束

            tState = np.clip(tState, states_min, states_max)
            for j in range(state_dim):
                tState[0][j] = (tState[0][j] - states_min[0][j])/(states_max[0][j] - states_min[0][j])

    log_f.close()