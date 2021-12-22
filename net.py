'''
Author: wbs2788
Date: 2021-12-20 20:13:11
LastEditTime: 2021-12-21 13:13:21
LastEditors: wbs2788
Description: 
FilePath: \Farmer-in-Minecraft\net.py

'''
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
#参数
BATCH_SIZE = 32
LR = 0.01                   # 学习率
EPSILON = 0.9               # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 4   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 2000      # 记忆库大小
N_ACTIONS = 7  # 棋子的动作0，1，2，3
N_STATES = 1

#神经网络

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self, n_f):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
        nn.Conv2d(n_f, n_f, 3, 1, 1), 
        nn.BatchNorm2d(n_f),
        nn.ReLU(),
        nn.Conv2d(n_f, n_f, 3, 1, 1),
        nn.BatchNorm2d(n_f),
        )

    def forward(self, x):
        x = x + self.residual(x)
        x = F.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__() 
        self.p1 = nn.MaxPool2d(3)       
        #self.c1 = nn.Conv2d(1, 8, 5, 1, 0)
        #self.c2 = nn.Conv2d(8, 4, 9, 1, 0)
        #self.c3 = nn.Conv2d(32, 64, 7, 1, 0)
        #self.c4 = nn.Conv2d(64, 32, 5, 1, 0)
        self.f1 = nn.Linear(49, 16)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2=nn.Linear(16, num_actions)
        self.f2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.p1(x)
        #x=self.c1(x)
        #x=F.relu(x)
        #x=self.c2(x)
        #x=F.relu(x)
        #x=self.c3(x)
        #x=F.relu(x)
        #x=self.c4(x)
        #x=F.relu(x)
        x=x.view(x.size(0),-1)
        #print(x.shape)
        x=self.f1(x)
        x=F.relu(x)   
        action=self.f2(x)
        return action

class DQN(object):
    def __init__(self):
        if torch.cuda.is_available():
            self.eval_net, self.target_net = Net().cuda(), Net().cuda() #DQN需要使用两个神经网络
        else:
            self.eval_net, self.target_net = Net(), Net()
        #eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0 # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0 # 记忆库记数
        self.memory = list(np.zeros((MEMORY_CAPACITY, 4))) # 初始化记忆库用numpy生成一个(2000,4)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, x, eps=EPSILON):
        if torch.cuda.is_available():
            x = torch.unsqueeze(torch.FloatTensor(x).cuda(), 0)
        else:
            x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample,x为场景
        if np.random.uniform() < eps:   # 选最优动作
            actions_value = self.eval_net.forward(x) #将场景输入Q估计神经网络
            #torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            if torch.cuda.is_available():
                action = torch.max(actions_value, 1)[1].data.cpu().numpy() # 返回动作最大值
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()
        else:   # 选随机动作
            action = np.array([np.random.randint(0, N_ACTIONS)]) # 比如np.random.randint(0,2)是选择1或0
        return action

    def store_transition(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s,a,r,s_]
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每100次
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s=[]
        b_a=[]
        b_r=[]
        b_s_=[]
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append(np.array(self.memory[i][1],dtype=np.int32))
            b_r.append(np.array([self.memory[i][2]],dtype=np.int32))
            b_s_.append(self.memory[i][3])
        if torch.cuda.is_available():
            b_s = torch.FloatTensor(b_s).cuda()#取出s
            b_a = torch.LongTensor(b_a).cuda() #取出a
            b_r = torch.FloatTensor(b_r).cuda() #取出r
            b_s_ = torch.FloatTensor(b_s_).cuda() #取出s_
        else:
            b_s = torch.FloatTensor(b_s)
            b_a = torch.LongTensor(b_a)
            b_r = torch.FloatTensor(b_r)
            b_s_ = torch.FloatTensor(b_s_)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)                          # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()                             # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0].reshape(BATCH_SIZE, 1)    # shape (batch, 1) DQL核心公式
        loss =  self.loss_func(q_eval, q_target) #计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad() #
        loss.backward() #反向传递
        self.optimizer.step()
        return loss