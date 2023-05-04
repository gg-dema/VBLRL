from bnn import BNN
from cem_optimizer import CEM_opt
import numpy as np

class Planner:
    def __init__(self,
                 stochastic_dyna: BNN,
                 action_dim=4,
                 plan_horizon=20,
                 num_particles=50,
                 num_sequence_action=20,
                 ):


        self.dynamic = stochastic_dyna
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_particles = num_particles
        self.num_sequence_action = num_sequence_action
        self.cem = CEM_opt(num_sequence_action)


    def _rollout_funct(self, actions):
        # calc 1/time_horizon * sum(reward for each action)
        reward = np.zeros(len(actions))
        for idx, a in enumerate(actions):
            # think about torch.no grad
            _, r = self.dynamic.forward(a)
            reward[idx] = r
        return np.mean(reward)

    def plan_step(self, state):
        # create particle st ---> nope, just re-run
        action_sequences = self.cem.sample_act()
        rewards = np.zeros(len(action_sequences))
        for idx, seq in enumerate(action_sequences):
            reward_of_particle = []
            for particle in range(self.num_particles):
                reward_of_particle.append(self.propagate(seq)) # return avg reward for the sequence
            rewards[idx] = np.mean(reward_of_particle)
        self.cem.update(rewards)
        # return best action : should be this, second option. To upgrade later
        return self.cem.solution()
