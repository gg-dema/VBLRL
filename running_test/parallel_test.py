
from bnn import BNN
from threading import Thread
from cem_optimizer import CEM_opt
import time
import torch


def propagate(sampled_linear, seq_act, state, reward_stock, index):
    r_tot = 0
    obs_shape = state.shape[0]
    for act in seq_act:
        x = torch.concatenate((state, torch.from_numpy(act)))
        y = sampled_linear(x)
        state, reward = y[:obs_shape], y[-1]
        r_tot += reward.item()
    reward_stock[index] = r_tot/seq_act.shape[0]


torch.set_default_dtype(torch.float64)
s = torch.randn((39, ))

stupid_nn = BNN(action_dim=4, obs_dim=s.shape[0], reward_dim=1)

cem = CEM_opt(population_shape=4*20,   # action dim * task horizon
              num_population=500)
seq_act = cem.sample_act()
seq_act = [seq_act[i].reshape((-1, 4)) for i in range(len(seq_act))]
particle_dynamics = [stupid_nn.sample_linear_net_functional('cpu') for _ in range(len(seq_act))]


rewards_for_action_seq = [None]*len(seq_act)

time_start = time.time()
for i in range(len(seq_act)):
        t = Thread(target=propagate,
                   args=(particle_dynamics[i],
                         seq_act[i],
                         s,
                         rewards_for_action_seq,
                         i))
        t.start()
        t.join()
print(f'time for rollout {time.time() - time_start}')

print(rewards_for_action_seq)
print(len(rewards_for_action_seq))

