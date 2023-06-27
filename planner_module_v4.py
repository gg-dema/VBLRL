from bnn import BNN
from cem_optimizer_v2 import CEM_opt
from Propagation import Propagation_net
import torch

_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Planner:
    def __init__(self,
                 stochastic_dyna: BNN,
                 plan_horizon=20,
                 num_particles=500,
                 percent_elite=0.1,
                 env_action_space_shape=4,
                 env_obs_space_shape=39,
                 device = 'cuda:0'
                 ):
        self.device = device

        self.dynamic = stochastic_dyna
        self.plan_horizon = plan_horizon

        self.num_particles = num_particles
        self.num_sequence_action = num_particles

        self.env_action_space_shape = env_action_space_shape
        self.env_obs_space_shape = env_obs_space_shape

        self.cem = CEM_opt(num_action_seq=num_particles,
                           action_seq_len=env_action_space_shape*plan_horizon,
                           percent_elite=percent_elite)
        self.propagation_net = Propagation_net(
            num_particles=num_particles,
            action_dim=env_action_space_shape,
            obs_dim=env_obs_space_shape
        )
        if self.device == 'cuda:0':
            self.propagation_net.move_to_gpu()




    def plan_step(self, state):

        action_sequences = torch.from_numpy(self.cem.population).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        rewards = torch.zeros(action_sequences.shape[0])

        for seq_idx, act_seq in enumerate(action_sequences):
            self.propagation_net.sample_from(self.dynamic)
            rewards[seq_idx] = self.propagation_net.propagate(state, act_seq).cpu().detach()

        self.cem.update(rewards.cpu().numpy())

        return self.cem.solutions()


if __name__ == '__main__':
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
    import time

    torch.set_default_dtype(torch.float64)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    class_env = "door-open-v2-goal-observable"
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[class_env]
    env = door_open_goal_observable_cls()

    action_space_shape = env.action_space.shape[0]
    obs_space_shape = env.observation_space.shape[0]

    s = env.reset()
    dyna = BNN(action_space_shape,
               obs_space_shape,
               reward_dim=1).to(device)
    planner = Planner(stochastic_dyna=dyna,
                      num_particles=500)

    start = time.time()
    print(planner.plan_step(s))
    print(f'total: {time.time() - start}')
