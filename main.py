
from buffers import MultiEnvReplayBuffer
from bnn import BNN
from lion_opt import Lion
from planner_module_v2 import Planner
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)


import numpy as np
import torchbnn as bnn
import torch.nn as nn

import random
import torch
import json

''' 
This main is project for run with the MetaWorld benchmark suite, 
for any migration to new benchmark, modify the bnn structure
'''




# for maximum compatibility with numpy array
torch.set_default_dtype(torch.float64)


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


def save_json_update(json_obj):
    with open('config_parameters.json', 'w') as f:
        json.dump(json_obj, f)

def json_load():
    with open('config_parameters.json', 'r') as f:
        config_file = json.load(f)
    return config_file

def add_const_to_namespace(config: dict):
    for k, v in config.items():
        globals()[k] = v

# ----------------------------------------------
#           CONFIG
# ----------------------------------------------

config = json_load()

# nested struct
buff_config = config['buffer']
planner_config = config['planner']
# just set of variable
const_config = config['const']
train_config = config['train']

add_const_to_namespace(const_config)
add_const_to_namespace(train_config)

# ----------------------------------------------
#           ENVS CREATION
# ----------------------------------------------
envs_name = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
envs = {}

for name in random.sample(envs_name, 10):
    if not buff_config["correspondence_id2env"].get(name, 0):
        buff_config["correspondence_id2env"][name] = buff_config["correspondence_id2env"]["first_idx_free"]
        buff_config["correspondence_id2env"]["first_idx_free"] += 1

    envs[buff_config["correspondence_id2env"][name]] = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]()

print(f'Sampled env: {list(envs.keys())}')


# ----------------------------------------------
#           MODEL INIT
# ----------------------------------------------

world_model = BNN(action_dim=ACTION_SHAPE,
                  obs_dim=OBS_SHAPE,
                  reward_dim=REWARD_SHAPE
                  ).to(DEVICE)

task_specific_models = {}
planners = {}
optimizers = {}
outer_opt = Lion(world_model.parameters(), lr=LR_OUTER)

for idx in envs.keys():
    task_specific_models[idx] = BNN(action_dim=ACTION_SHAPE,
                                    obs_dim=OBS_SHAPE,
                                    reward_dim=REWARD_SHAPE,
                                    weight_world_model=world_model.state_dict()
                                    ).to(DEVICE)
    if LOAD_OLD_MODEL:
        print('non ho ancora alcun modello, scemo')


    #TODO: save or load the planner
    planners[idx] = Planner(stochastic_dyna=task_specific_models[idx],
                            action_dim=ACTION_SHAPE,
                            plan_horizon=planner_config['plan_horizon'],
                            num_particles=planner_config['num_particles'],
                            num_elite=planner_config['cem']['num_elite'],
                            num_sequence_action=planner_config['cem']['population'])

    optimizers[idx] = Lion(task_specific_models[idx].parameters(),
                           lr=LR_INNER)


# ----------------  loss  ----------------------
mse_loss = nn.MSELoss().to(DEVICE)
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(DEVICE)
# ----------------------------------------------
#           BUFFER INIT
# ----------------------------------------------

multibuffer = MultiEnvReplayBuffer(
    buffer_size_per_env=buff_config['max_size_for_env']
)
if buff_config['preload']:
    multibuffer.read_buffers(buff_config['IO_option']['path'],
                             buff_config['IO_option']['from_scratch']
                             )
for k in envs.keys():

    if multibuffer.elem_for_buffer[k] == 0:
        #if not any sample for that exp: do random move
        print(f'adding random step for model {k}')
        s = envs[k].reset()
        for _ in range(500):
            s = envs[k].reset()
            act = envs[k].action_space.sample()
            new_s, reward, done, info = envs[k].step(act)
            multibuffer.add(
                s, act, reward,
                new_s, done,  env_id=k
            )
multibuffer.write_buffer(buff_config['IO_option']['path'])

true)
    kl = kl_loss(net)
    loss = mse + KL_WEIGHT*kl
    loss.backward()
    optimizer.step()
    return mse.item(), kl.item(), loss.item()


def save_json_update(json_obj):
    with open('config_parameters.json', 'w') as f:
        json.dump(json_obj, f)

def json_load():

# ----------------------------------------------
#           RUN
# ----------------------------------------------
for k in envs.keys():
    for ep in range(EPISODE_FOR_TASK):  # default val : 100
        actual_state = envs[k].reset()
        actual_state = actual_state.astype(np.float32)
        reward_for_ep = []
        for t in range(planner_config['plan_horizon']):  # default value plan horizon: 20
            best_action = planners[k].plan_step(actual_state)
            new_state, reward, done, info = envs[k].step(best_action)
            multibuffer.add(best_action,
                            actual_state,
                            reward,
                            new_state,
                            done,
                            env_id=k)
            actual_state = new_state
            reward_for_ep.append(reward)

            loss = update_with_elbo(planners[k].dynamic,
                                    multibuffer,
                                    optimizer=optimizers[k],
                                    task_id=k)
            print(f'ep {ep} | plan_step: {t} | [mse, kl, tot_loss] {loss} | reward {reward}')

            if done:
                break
        multibuffer.write_buffer('VBLRL_rl_exam/buffer_stock')
        print(f'#--------- avg rew for ep {np.mean(reward_for_ep)}')
        # for now there is no meaning in train the world model
        update_with_elbo(world_model, multibuffer, optimizer=outer_opt, task_id=-1)
