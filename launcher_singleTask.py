from buffers import MultiEnvReplayBuffer
from bnn import BNN
from lion_opt import Lion
from planner_module_v2 import Planner
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)


import numpy as np
import torchbnn as bnn
import torch.nn as nn
import torch

# just for compatibility with numpy (default float : 64)
# it's unecessary, 32 bit are enough, maybe 16 to ---> faster
torch.set_default_dtype(torch.float64)

class_env = "door-open-v2-goal-observable"
door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[class_env]
env = door_open_goal_observable_cls()
s = env.reset()

action_space_shape = env.action_space.shape[0]
obs_space_shape = env.observation_space.shape[0]
PLAN_HORIZON = 20
EPISODE_FOR_TASK = 100
BATCH_SIZE = 256
KL_WEIGHT = 0.01
PRELOAD_BUFFER = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def update_with_elbo(net, buffer, optimizer, task_id):
    states, actions, rewards, next_states, dones = buffer.sample_all_envs(batch_size=BATCH_SIZE) \
        if (task_id == -1) \
        else buffer.sample_env(task_id, batch_size=BATCH_SIZE)

    optimizer.zero_grad()
    x = torch.Tensor(np.concatenate((states, actions), axis=-1)).to(DEVICE)
    y_true = torch.Tensor(np.concatenate((next_states, rewards), axis=-1)).to(DEVICE)
    y_pred = net(x)

    mse = mse_loss(y_pred, y_true)
    kl = kl_loss(net)
    loss = mse + KL_WEIGHT*kl
    loss.backward()
    optimizer.step()
    return mse.item(), kl.item(), loss.item()

buffer = MultiEnvReplayBuffer(
    buffer_size_per_env=1000
)

# fill buffer for random act
if not PRELOAD_BUFFER:
    for _ in range(500):
        act = env.action_space.sample()
        new_s, reward, done, info = env.step(act)
        buffer.add(s, act, reward, new_s, done, env_id=0)
else:
    buffer.read_buffer("VBLRL_rl_exam/buffer_stock")


# ----------------------------------------------
#   WORLD MODEL
# ----------------------------------------------



world_model = BNN(action_dim=action_space_shape,
                  obs_dim=obs_space_shape,
                  reward_dim=1,
                  ).to(DEVICE)
# ----------------------------------------------
#   TASK SPECIFIC MODEL
# ----------------------------------------------
task_specific = BNN(action_dim=action_space_shape,
                    obs_dim=obs_space_shape,
                    reward_dim=1,
                    weight_world_model=world_model.state_dict()).to(DEVICE)

mse_loss = nn.MSELoss().to(DEVICE)
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(DEVICE)
inner_optimizer = Lion(task_specific.parameters(), lr=1e-3)
outer_optimizer = Lion(world_model.parameters(), lr=1e-2)


planner = Planner(stochastic_dyna=task_specific,
                  action_dim=action_space_shape,
                  plan_horizon=PLAN_HORIZON,
                  num_particles=80,
                  num_elite=30,
                  num_sequence_action=20)

# training loop
task_id = 0
for ep in range(EPISODE_FOR_TASK):
    actual_state = env.reset()
    actual_state = actual_state.astype(np.float32)
    reward_for_ep = []
    for t in range(PLAN_HORIZON):
        best_action = planner.plan_step(actual_state)
        new_state, reward, done, info = env.step(best_action)
        buffer.add(actual_state,
                   best_action,
                   reward,
                   new_state,
                   done, env_id=0)
        actual_state = new_state
        reward_for_ep.append(reward)

        loss = update_with_elbo(planner.dynamic, buffer, optimizer=inner_optimizer, task_id=task_id)
        print(f'ep {ep} | plan_step: {t} | [mse, kl, tot_loss] {loss} | reward {reward}')

        if done:
            break
    buffer.write_buffer('VBLRL_rl_exam/buffer_stock')
    print(f'#--------- avg rew for ep {np.mean(reward_for_ep)}')
    # for now there is no meaning in train the world model
    # update_with_elbo(world_model, buffer, optimizer=outer_optimizer, task_id=-1)
