import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
env = door_open_goal_observable_cls()
env.reset()
action = np.array([0, 0.2, 0.1, 0])
while True:
    env.step(np.random.randn(4))
    env.render()
