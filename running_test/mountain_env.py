import math

import gym
import numpy as np

from cem_optimizer_v2 import CEM_opt
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('MountainCarContinuous-v0')
s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]


class Agent(nn.Module):

    def __init__(self):
        super().__init__()
        self.s_size = s_size
        self.a_size = a_size

        self.fc1 = nn.Linear(self.s_size, 100)
        self.fc2 = nn.Linear(100, self.a_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x.cpu().data

def act(agent_tmp, state):
    state = torch.from_numpy(state)
    with torch.no_grad():
        action = agent_tmp(state)
    return action

def reconstruct_dict(vector, reconstruct_dict):
    index = 0
    params_recostruct = {}
    for k, v in reconstruct_dict.items():
        temp = vect[index:index+v]
        if k == 'fc1.weight': temp = temp.reshape((100,2))
        if k == 'fc1.bias': temp = temp.reshape((100,))
        if k == 'fc2.weight' : temp = temp.reshape((1,100))
        if k == 'fc2.bias' : temp = temp.reshape((1))
        params_recostruct[k] = torch.from_numpy(temp)
        index =+v
    return params_recostruct

def divide_dict(state_dict):
    reconstruct_dict = {}
    vect = np.array([])
    for k, v in params.items():
        v = v.detach()
        v = v.reshape(-1)
        numb_elem = v.shape[0]
        reconstruct_dict[k] = numb_elem
        vect = np.concatenate((vect, v), axis=-1)
    return reconstruct_dict, vect

agent = Agent()
params = dict(agent.named_parameters())
rec_dict, vect = divide_dict(params)
params_reborn = reconstruct_dict(vect, rec_dict)
agent2 = Agent()
agent2.load_state_dict(params_reborn)

cem = CEM_opt(population_shape=len(vect),
              num_population=500
)


for i in range(2000):

    possible_vect = cem.sample_act()
    reward = []
    for sol in possible_vect:
        params_reborn = reconstruct_dict(sol, rec_dict)
        agent.load_state_dict(params_reborn)
        r_sol = 0
        s, _ = env.reset()
        for horizon in range(20):
            action = act(agent, s)
            new_s, r, d, _, _ = env.step(action)
            if d: break
            r_sol += r*math.pow(0.90, horizon)
        reward.append(r_sol)
    cem.update(reward)
    if i%10==0:
        print(i, ':', np.mean(reward))


