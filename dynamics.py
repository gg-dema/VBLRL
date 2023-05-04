from bnn import BNN
import metaworld
import random

class dynamics:

    def __init__(self, num_task):
        self.world_model = BNN()
        self.task_specific_models = []
        for i in range(num_task):
            self.task_specific_models.append(BNN())

    def init_task_specific(self, task_index):
        self.task_specific_models[task_index].init_weight_from(self.world_model)

class fake_dynamics:

    def __init__(self):
        ml1 = metaworld.ML1('pick-place-v2')
        task = random.choice(ml1.train_tasks)
        self.env = ml1.train_classes['pick-place-v2']()
        self.env.set_task(task)

    def forward(self, action):
        state_prime, reward, done, info = self.env.step(action)
        return state_prime, reward
