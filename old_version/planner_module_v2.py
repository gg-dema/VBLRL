from bnn import BNN
from cem_optimizer import CEM_opt
import numpy as np
import torch

_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class Planner:
    def __init__(self,
                 stochastic_dyna: BNN,
                 action_dim=4,
                 plan_horizon=20,
                 num_particles=50,
                 num_sequence_action=50,
                 num_elite=30,
                 env_action_space_shape=4,
                 env_obs_space_shape=39,

                 ):

        self.dynamic = stochastic_dyna
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_particles = num_particles
        self.num_sequence_action = num_sequence_action
        self.env_action_space_shape = env_action_space_shape
        self.env_obs_space_shape = env_obs_space_shape

        # NOT Sure on pop shape
        self.cem = CEM_opt(population_shape=(action_dim*plan_horizon),
                           num_population=num_sequence_action,
                           num_elite=num_elite)

    def _rollout_funct(self, state, actions):
        # calc 1/time_horizon * sum(reward for each action)
        reward = np.zeros(len(actions))
        state = torch.from_numpy(state).to(_DEVICE)

        actions = actions.reshape((-1, 4))
        self.dynamic.deterministic_mode()
        for idx, a in enumerate(actions):
            with torch.no_grad():
                a = torch.from_numpy(a).to(_DEVICE)
                x = torch.concatenate((state, a))
                y = self.dynamic.forward(x)
                state, r = y[:len(y)-1], y[-1]
                # problem : how can the reward be = 0 for the end of the task?
            reward[idx] = r
        self.dynamic.stochatisc_mode()
        return np.mean(reward)



    #@timebudget
    def plan_step(self, state):
        # create particle st ---> nope, just re-run
        action_sequences = self.cem.sample_act()
        rewards = np.zeros(len(action_sequences))
        for idx, seq in enumerate(action_sequences):
            reward_of_particle = []
            for particle in range(self.num_particles):
                reward_of_particle.append(self._rollout_funct(state, seq))  # return avg reward for the sequence
            rewards[idx] = np.mean(reward_of_particle)
        self.cem.update(rewards)
        # return best action : should be this, second option. To upgrade later
        return self.cem.solution[:self.env_action_space_shape]



if __name__ == '__main__':
    # scope of this: just testing
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
    import time

    torch.set_default_dtype(torch.float64)
    class_env = "door-open-v2-goal-observable"
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[class_env]
    env = door_open_goal_observable_cls()

    action_space_shape = env.action_space.shape[0]
    obs_space_shape = env.observation_space.shape[0]

    s = env.reset()
    dynam = BNN(action_space_shape,
                obs_space_shape,
                reward_dim=1).to(_DEVICE)
    planner = Planner(dynam,
                      action_dim=action_space_shape,
                      plan_horizon=20,
                      num_particles=30,
                      num_sequence_action=200)

    # piu particelle : migliore stima della Q funct
    # maggior num seq action : maggior varieta' nella stima delle azioni possibili (?)
    # plan_horizon :
    start = time.time()
    print(planner.plan_step(s))
    print(f'total: {time.time() - start}')
