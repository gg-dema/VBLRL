import numpy as np

class CEM_opt:
    '''
    gradient free optimizer
    We want to use it for get the action distribution
    QUESTION: HOW IS RELATED TO THE ORIGINAL STATE?
    TEMPORAL ANSWER: is not, for now is just the best distrib for a task specific
    '''

    def __init__(self,
                 population_shape,   # action dim * task horizon
                 num_population: int,
                 num_elite: int = 30):
        self.num_elite = num_elite
        self.num_population = num_population
        self.population_shape = population_shape
        self.mean_vect = np.zeros(self.population_shape)
        self.min = -1.
        self.max = 1.
        self.population = [
            self.mean_vect + np.random.rand(self.population_shape)*0.8
            for _ in range(self.num_population)
        ]

    def update(self, rewards: np.array):
        elite_idxs = np.array(rewards).argsort()[-self.num_elite:]
        elite_weights = [self.population[idx] for idx in elite_idxs]
        self.mean_vect = np.array(elite_weights).mean(axis=0)

        self.population = [
            np.clip(self.mean_vect + np.random.rand(self.population_shape)*0.8, self.min, self.max)
            for _ in range(self.num_population)
        ]


    @property
    def solution(self):
        return self.mean_vect

    def sample_act(self):
        return self.population
