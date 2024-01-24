import torch
import torch.nn as nn
from torch.nn import init
from torch.distributions import MultivariateNormal  # 多元正态分布
from torch.distributions import Categorical # 可以按照一定概率产生具体数字
import numpy as np

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space  # 动作空间是否连续

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)   # 为变量赋同样的值（标准差的平方，方差）

        # actor
        if has_continuous_action_space :    # 连续
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),  # 非线性激活函数
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        ) # 简单的顺序连接
        else:   # 离散
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)  # 归一化函数，元素之和为1；dim，维数，矩阵层数
                        )


        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):   # 只存在于连续动作空间

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError   # 如果这个方法没有被子类重写，但是调用了，就会报错


    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            # self.action_var[0] = self.action_var[1]/2     # 2023-2-20
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 先变成对角矩阵，然后再扩充维数，1*action_dim*action_dim
            dist = MultivariateNormal(action_mean, cov_mat) # 按照均值和方差确定分布
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)    # 按照网络输出概率分布动作分布，整数

        action = dist.sample()  # 采样得到动作
        action_logprob = dist.log_prob(action)  # 返回概率的log
        
        return action.detach(), action_logprob.detach() # 切断一些分支的反向传播
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean) # 把一个tensor变成和函数括号内一样形状的tensor
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action) # 返回行动概率的log
        dist_entropy = dist.entropy()   # 返回分布的熵（混乱程度）
        state_values = self.critic(state)   # 对状态打分
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lamda, K_epochs, batch_size, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space  # 是否连续

        if has_continuous_action_space:
            self.action_std = action_std_init   # 标准差

        self.gamma = gamma  # 折扣因子
        self.lamda = lamda
        self.eps_clip = eps_clip    # 裁剪参数
        self.K_epochs = K_epochs    # 策略更新间隔
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()   # buffer

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)   # agent
        # for name, param in self.policy.actor.named_parameters():
        #     if 'weight' in name:
        #         init.orthogonal_(param, gain=1.41) # 正交初始化 2022-9-28
        #     if '4.weight' in name:
        #         init.orthogonal_(param, gain=1.41 * 0.01)
        #     print(name, param.data)
        # for name, param in self.policy.critic.named_parameters():
        #     if 'weight' in name:
        #         init.orthogonal_(param, gain=1.41) # 正交初始化。目前看来没有用

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])  # 优化器 2022-9-28

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)   # target网络
        self.policy_old.load_state_dict(self.policy.state_dict())   # 加载同样的参数
        
        self.MseLoss = nn.MSELoss() # 均方损失函数

        self.steer_range = (-0.8, 0.8)
        self.throttle_range = (0.6, 1.0)

        self.ob_rms = RunningMeanStd(shape=state_dim) # 状态归一化

    def set_action_std(self, new_action_std):   # 加载新的标准差
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):  # 标准差衰减
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate   # 线性衰减
            # self.action_std = self.action_std * 0.95    # 指数衰减
            self.action_std = round(self.action_std, 4) # 指定小数点位数
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():   # 不会被反向传递
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            steer = float(torch.tanh(action[0, 0]).detach().cpu().numpy())
            throttle = float(torch.tanh(action[0, 1]).detach().cpu().numpy())

            steer = (steer + 1) / 2 * (self.steer_range[1] - self.steer_range[0]) + self.steer_range[0]
            throttle = (throttle + 1) / 2 * (self.throttle_range[1] - self.throttle_range[0]) + self.throttle_range[0]

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            print("actor output action_std: ", self.action_std)

            return np.array([steer, throttle])  # 随机

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)   # 奖励衰减
            rewards.insert(0, discounted_reward)    # 首插入
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device) # 奖励标准化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)  # 张量转化
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)    # 去掉维数是1的，
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        old_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        logprobs, state_values, dist_entropy = self.policy_old.evaluate(old_states, old_actions)
        # GAE
        advantages = []
        gae = 0
        for i in reversed(range(len(old_rewards))):
            mask = 1.0 - old_terminals[i]
            if i == len(old_rewards) - 1:
                delta = old_rewards[i] - state_values[i]
            else:
                delta = old_rewards[i] + self.gamma * state_values[i + 1] * mask - state_values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device) # 奖励标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):  # 迭代优化K次

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)    # 评估阶段

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy   # 计算损失
            
            # take gradient step
            self.optimizer.zero_grad()  # 将梯度初始化为零
            loss.mean().backward()  # 反向传播求梯度
            self.optimizer.step()   # 更新参数
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())   # 权重赋值

        # clear buffer
        self.buffer.clear() # buffer清零
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count