from buffers import MultiEnvReplayBuffer
from bnn import BNN
from lion_opt import Lion
from planner_module_v2 import Planner
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)


class_env = "door-open-v2-goal-observable"
door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[class_env]
env = door_open_goal_observable_cls()
s = env.reset()

action_space_shape = env.action_space.shape[0]
obs_space_shape = env.observation_space.shape[0]
PLAN_HORIZON = 50
EPISODE_FOR_TASK = 100



def update_with_elbo(net, buffer, task_id):
    print('ti piacerebbe')


buffer = MultiEnvReplayBuffer(
    buffer_size_per_env=1000
)

# fill buffer for random act

for _ in range(100):
    act = env.action_space.sample()
    new_s, reward, done, info = env.step(act)
    buffer.add(s, act, reward, new_s, done, env_id=0)

# buffer.write_buffer('VBLRL_rl_exam/buffer_stock')


# ----------------------------------------------
#   WORLD MODEL
# ----------------------------------------------

world_model = BNN(action_dim=action_space_shape,
                  obs_dim=obs_space_shape,
                  reward_dim=1,
                  ).double()
# preload weight : TODO
outer_optimizer = Lion(world_model.parameters())


# ----------------------------------------------
#   TASK SPECIFIC MODEL
# ----------------------------------------------
# weight = same of world model
task_specific = BNN(action_dim=action_space_shape,
                    obs_dim=obs_space_shape,
                    reward_dim=1,
                    W_world_model=world_model.state_dict()).double()

# TO CHECK : should i optimize the task model or the planner ?
inner_optimizer = Lion(task_specific.parameters())

planner = Planner(stochastic_dyna=task_specific,
                  action_dim=action_space_shape,
                  plan_horizon=PLAN_HORIZON,
                  num_particles=50,
                  num_sequence_action=50)


# training loop
task_id = 0
for ep in range(EPISODE_FOR_TASK):
    actual_state = env.reset()

    for t in range(PLAN_HORIZON):
        best_action = planner.plan_step(actual_state)
        new_state, reward, done, info = env.step(best_action)
        buffer.add(actual_state,
                   best_action,
                   reward,
                   new_state,
                   done)
        actual_state = new_state

        update_with_elbo(planner.dynamic, buffer, task_id=task_id)
        if done: break
    update_with_elbo(world_model, buffer, task_id=-1)


