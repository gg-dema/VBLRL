import buffers
from bnn import BNN
from planner_module_v2 import Planner
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
from lion_opt import Lion



def update_with_elbo(buffer, net, optimizer):



# const
PLAN_HORIZON = 20
EPISODE_FOR_TASK = 100



# environment definition
door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
env = door_open_goal_observable_cls()
env.reset()

action_space_shape = env.action_space.shape[0]
obs_space_shape = env.observation_space.shape[0]


#buffer = MultiEnvReplayBuffer(...)
buffer = buffers.SingleTaskReplayBuffer()
world_model = BNN(action_dim=action_space_shape,
                  obs_dim=obs_space_shape,
                  reward_dim=1,
                  )  # ---> load weight (12-maggio, primo train, non ho pesi ancora)

outer_optimizer = Lion(world_model.parameters())

# ----------------------------------------------
#-----------------------------------------------

task_specific = BNN(action_dim=action_space_shape,
                    obs_dim=obs_space_shape,
                    reward_dim=1,
                    W_world_model=world_model.state_dict())

# for task in task_list
# fill buffer: random mode ---> epsilon greedy init
# should be part of replay buffer modules?

planner = Planner(stochastic_dyna=task_specific,
                  action_dim=action_space_shape,
                  plan_horizon=PLAN_HORIZON,
                  num_particles=50,
                  num_sequence_action=50)

inner_optimizer = Lion(task_specific.parameters())


# training
for ep in range(EPISODE_FOR_TASK):
    actual_state = env.reset()

    for t in range(PLAN_HORIZON):
        # the planner module encapsulate the bnn
        best_action = planner.plan_step(actual_state)
        new_state, reward, done, info = env.step(best_action)
        # add all to replay buffer
        # update elbo task specific
        update_with_elbo(planner.dynamic, buffer, task_id=1)
        if done:
            break

    update_with_elbo(world_model, buffer, task_id=0)










