from buffers import MultiEnvReplayBuffer
from bnn import BNN
from lion_opt import Lion
from planner_module_v2 import Planner
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
import numpy as np
import torchbnn as bnn
import torch.nn as nn
import torch
import pickle
import json
import random
import os

''' 
This main is project for run with the MetaWorld benchmark suite, 
for any migration to new benchmark, modify the bnn structure
'''

# for maximum compatibility with numpy array
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def update_with_elbo(net, buffer, optimizer, task_id):
    states, actions, rewards, next_states, dones = buffer.sample_all_envs(batch_size=BATCH_SIZE) if (
                task_id == -1) else buffer.sample_env(task_id, batch_size=BATCH_SIZE)

    optimizer.zero_grad()
    x = torch.Tensor(np.concatenate((states, actions), axis=-1)).to(DEVICE)
    y_true = torch.Tensor(np.concatenate((next_states, rewards), axis=-1)).to(DEVICE)
    y_pred = net(x)

    mse = mse_loss(y_pred, y_true)
    kl = kl_loss(net)
    loss = mse + KL_WEIGHT * kl
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


def save_torch_models(models: dict, path,):
    for k, model in models.items():
        name = os.path.join(path, f'model_env{k}.pth')
        torch.save(model.state_dict(), name)


def save_binary_objs(objs, path, name_template: str):
    for k, obj in objs.items():
        name = os.path.join(path, name_template+f'{k}.pkl')
        with open(name, 'wb') as f:
            pickle.dump(obj, f)


def load_binary_file(path):
    with open(path, 'rb') as f:
        obj = pickle.load(path)
    return obj


if __name__ == '__main__':
    """
    ----------------------------------------------
               CONFIG
    ----------------------------------------------
    """
    config = json_load()

    # nested struct
    buff_config = config['buffer']
    planner_config = config['planner']
    # just set of variable
    const_config = config['const']
    train_config = config['train']

    # i load only the var that i need inside the
    # traininig loop, not in the initialization of the
    # object. If each time i should look inside the dict,
    # i'll consume operation

    DEVICE = const_config['DEVICE']
    BATCH_SIZE = train_config['BATCH_SIZE']
    KL_WEIGHT = train_config['KL_WEIGHT']
    EPISODE_FOR_TASK = train_config['EPISODE_FOR_TASK']
    """
    ----------------------------------------------
               ENVS CREATION
    ----------------------------------------------
            random select a subset of 10 envs from all possible envs
            in the config file we have a dict[name: idx], here we update all the index 
    """
    envs_name = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
    envs = {}
    for name in random.sample(envs_name, 3):
        if not buff_config["correspondence_id2env"].get(name, 0):
            buff_config["correspondence_id2env"][name] = buff_config["correspondence_id2env"]["first_idx_free"]
            buff_config["correspondence_id2env"]["first_idx_free"] += 1
        envs[buff_config["correspondence_id2env"][name]] = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]()

    config['buffer'] = buff_config
    save_json_update(config)

    """
    ----------------------------------------------
             BUFFER INIT
    ----------------------------------------------
            create a multi-env buffer, using the mapping from config.dict[name : index]
            we load all old transitions OR we do random step for collect new transitions (new env)
            
            small note: the update for the world should be done with all the buffer, so we should load all
    """

    multibuffer = MultiEnvReplayBuffer(
        buffer_size_per_env=buff_config['max_size_for_env']
    )
    # ----- load automatically all the buffer ------
    if buff_config['preload']:
        multibuffer.read_buffers(buff_config['IO_option']['path'],
                                 buff_config['IO_option']['from_scratch']
                                 )

    # ---- random action for empty buffer ---------
    for k in envs.keys():
        if multibuffer.elem_for_buffer[k] == 0:
            # if not any sample for that exp: do random move
            print(f'adding random step for model {k}')
            s = envs[k].reset()
            for _ in range(500):
                s = envs[k].reset()
                act = envs[k].action_space.sample()
                new_s, reward, done, info = envs[k].step(act)
                assert (s.shape == (39,))
                multibuffer.add(
                    s, act, reward,
                    new_s, done, env_id=k
                )
    # --------------- save new buffer ---------------
    multibuffer.write_buffer(buff_config['IO_option']['path'])

    """
    ----------------------------------------------
               MODEL INIT
    ----------------------------------------------
    """

    # ---------- world -----------------------------
    world_model = BNN(action_dim=const_config['ACTION_SHAPE'],
                      obs_dim=const_config['OBS_SHAPE'],
                      reward_dim=const_config['REWARD_SHAPE']
                      ).to(DEVICE)

    model_name = os.path.join(
        train_config['path_world_model'],
        'model_world.pth'
    )
    outer_opt = Lion(world_model.parameters(), lr=train_config['LR_OUTER'])

    if os.path.isfile(model_name) and train_config['LOAD_WORLD_MODEL']:
        world_model.load_state_dict(torch.load(model_name))
        outer_opt = load_binary_file(os.path.join(train_config['path_optimizer'], 'optimizerWorld.pkl'))

    # --------- task specific ----------------------

    task_specific_models = {}
    planners = {}
    optimizers = {}

    for idx in envs.keys():
        model_name = os.path.join(train_config['path_task_specific'], f'model_env{idx}.pth')
        planner_name = os.path.join(train_config['path_optimizer'], f'optimizer_env{idx}.pkl')
        if os.path.isfile(model_name) and train_config['LOAD_TASK_MODEL']:

            task_specific_models[idx] = BNN(const_config['ACTION_SHAPE'],
                                            obs_dim=const_config['OBS_SHAPE'],
                                            reward_dim=const_config['REWARD_SHAPE'],
                                            weight_world_model=torch.load(model_name)
                                            ).to(DEVICE)
        else:  # load params from world model
            task_specific_models[idx] = BNN(const_config['ACTION_SHAPE'],
                                            obs_dim=const_config['OBS_SHAPE'],
                                            reward_dim=const_config['REWARD_SHAPE'],
                                            weight_world_model=world_model.state_dict()
                                            ).to(DEVICE)

        # TODO: save or load the planner
        if os.path.isfile(planner_name) and train_config['LOAD_TASK_MODEL']:
            planners[idx] = load_binary_file(planner_name)
        else:
            planners[idx] = Planner(stochastic_dyna=task_specific_models[idx],
                                    action_dim=const_config['ACTION_SHAPE'],
                                    plan_horizon=planner_config['plan_horizon'],
                                    num_particles=planner_config['num_particles'],
                                    num_elite=planner_config['cem']['num_elite'],
                                    num_sequence_action=planner_config['cem']['population'])

        optimizers[idx] = Lion(task_specific_models[idx].parameters(),
                               lr=train_config['LR_INNER'])

    # ----------------  loss  ----------------------
    mse_loss = nn.MSELoss().to(DEVICE)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(DEVICE)
    """
    ----------------------------------------------
               RUN
    ----------------------------------------------
    """

    for k in envs.keys():
        for ep in range(EPISODE_FOR_TASK):  # default val : 100
            actual_state = envs[k].reset()
            reward_for_ep = []
            for t in range(planner_config['plan_horizon']):  # default value plan horizon: 20
                best_action = planners[k].plan_step(actual_state)
                new_state, reward, done, info = envs[k].step(best_action)
                multibuffer.add(actual_state,
                                best_action,
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
                print(f'env_id: {k} | ep {ep} | plan_step: {t} | [mse, kl, tot_loss] {loss} | reward {reward}')
                if done: break

            print(f'#--------- avg rew for ep {np.mean(reward_for_ep)}')
            # for now there is no meaning in train the world model
            update_with_elbo(world_model, multibuffer, optimizer=outer_opt, task_id=-1)
            if (ep % 10) == 0:
                multibuffer.write_buffer(buff_config['IO_option']['path'])
                save_binary_objs(planners, planner_config['path_old_planner'], name_template='planner_env')
                save_binary_objs(optimizers, train_config['path_optimizer'], name_template='optimizer_env')
                save_binary_objs({'World':outer_opt}, train_config['path_optimizer'], name_template='optimizer')
                save_torch_models({'World': world_model}, train_config['path_world_model'])
                save_torch_models(task_specific_models, train_config['path_task_specific'])
