from bnn import BNN
from cem_optimizer import CEM_opt
from threading import Thread
import numpy as np
import torch

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Planner:
    def __init__(
        self,
        stochastic_dyna: BNN,
        action_dim=4,
        plan_horizon=20,
        num_particles=50,
        num_elite=30,
        env_action_space_shape=4,
        env_obs_space_shape=39,
    ):
        self.dynamic = stochastic_dyna
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_particles = num_particles
        self.num_sequence_action = num_particles
        self.env_action_space_shape = env_action_space_shape
        self.env_obs_space_shape = env_obs_space_shape

        # NOT Sure on pop shape
        self.cem = CEM_opt(
            population_shape=(action_dim * plan_horizon),
            num_population=num_particles,
            num_elite=num_elite,
        )

    def plan_step(self, state):
        action_sequences = self.cem.sample_act()
        action_sequences = [
            action_sequences[i].reshape((-1, 4))
            for i in range(self.num_sequence_action)
        ]

        rewards_for_sequence = np.zeros(self.num_sequence_action)
        for idx, sequence in enumerate(action_sequences):
            rewards_for_sequence[idx] = self.eval_action_seq(state, sequence)
        self.cem.update(rewards_for_sequence)
        return self.cem.solution[: self.env_action_space_shape]

    def eval_action_seq(self, state, action_sequence):
        rewards = [None] * self.num_particles
        threads = []

        for i in range(self.num_particles):
            t = Thread(
                target=self.propagate,
                args=(
                    self.dynamic.sample_linear_net_functional("cpu"),
                    action_sequence,
                    state,
                    rewards,
                    i,
                ),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return sum(rewards) / self.num_particles

    def propagate(self, dynamic, action_seq, state, rewards_array, index):
        state = torch.from_numpy(state)
        r_tot = 0
        obs_shape = state.shape[0]
        for h, act in enumerate(action_seq):
            x = torch.concatenate((state, torch.from_numpy(act)))
            y = dynamic(x)
            state, reward = y[:obs_shape], y[-1] * (h**0.99)
            r_tot += reward.item()
        rewards_array[index] = r_tot / action_seq.shape[0]


if __name__ == "__main__":
    # scope of this: just testing
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    import time

    torch.set_default_dtype(torch.float64)
    class_env = "door-open-v2-goal-observable"
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[class_env]
    env = door_open_goal_observable_cls()

    action_space_shape = env.action_space.shape[0]
    obs_space_shape = env.observation_space.shape[0]

    s = env.reset()
    dynam = BNN(action_space_shape, obs_space_shape, reward_dim=1).to(_DEVICE)
    planner = Planner(
        dynam, action_dim=action_space_shape, plan_horizon=20, num_particles=500
    )
    # piu particelle : migliore stima della Q funct
    # maggior num seq action : maggior varieta' nella stima delle azioni possibili (?)
    # plan_horizon :
    start = time.time()
    print(planner.plan_step(s))
    print(f"total: {time.time() - start}")
