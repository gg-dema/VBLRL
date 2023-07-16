
from VBLRL_rl_exam.cem_optimizer_v2 import CEM_opt
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
import time
import torch


env_name = 'plate-slide-side-v2-goal-observable'
env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
env_planner = env_cls()
env_test = env_cls()

env_action_space_shape = env_test.action_space.shape[0]
env_obs_space_shape = env_test.observation_space.shape[0]


class InformedPlanner:

    def __init__(self, env):

        self.env = env
        self.horizon = 80
        self.num_sequence_action = 100
        self.cem = CEM_opt(num_action_seq=self.num_sequence_action,
                           action_seq_len=env_action_space_shape * self.horizon,
                           percent_elite=0.1)
        self.action_seq_planned = Queue(maxsize=self.horizon)


    def plan(self, force_replan=False):

        if self.action_seq_planned.empty() or force_replan:

            action_sequences = self.cem.population
            rewards = np.zeros(action_sequences.shape[0])
            for idx, seq in enumerate(action_sequences):
                rewards[idx] = self.eval_act_seq(seq)
            self.cem.update(rewards)

            for act in self.cem.solutions().reshape(-1, 4):
                self.action_seq_planned.put(act)
        return self.action_seq_planned.get()



    def eval_act_seq(self, sequence):
        rew_seq = 0
        self.env.reset()
        act_reshaped = sequence.reshape((-1, 4))
        for act in act_reshaped:

            _, r, _, _ = self.env.step(act)
            rew_seq += r
        return rew_seq/len(sequence)



planner = InformedPlanner(env_planner)

TOT_avg_rew = []
TOT_avg_rew.append(0)
TOT_avg_rew.append(0)

for ep in range(10):

    env_test.reset()

    r_for_ep = 0
    for h in range(80):
        act = planner.plan()
        s_prime, r, done, _ = env_test.step(act)
        r_for_ep += r
        if r==10.0:
            done= True
            print(f"SOLVED IN {ep} iteration")
            break
        print(f'episode {ep}, step {h}, reward  {r} | done? {done}')
    if done == True: break

    TOT_avg_rew.append(r_for_ep)
    print(f'prev avg rew {TOT_avg_rew[-2]}, actual avg rew {TOT_avg_rew[-1]}')


input('.... ready for register? ')

env_test.reset()
act_seq = planner.cem.solutions()
act_seq = act_seq.reshape((-1, 4))
for idx, a in enumerate(act_seq):
    env_test.render()
    env_test.step(a)
env_test.close()

