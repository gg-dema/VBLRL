import metaworld
import random


class fake_dynamics:

    def __init__(self):
        ml1 = metaworld.ML1('pick-place-v2')
        task = random.choice(ml1.train_tasks)
        self.env = ml1.train_classes['pick-place-v2']()
        self.env.set_task(task)

    def forward(self, action):
        return_val = self.env.step(action)
        next_state, reward = return_val[:2]
        return next_state, reward
