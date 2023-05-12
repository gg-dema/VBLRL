import numpy as np

class CEM_opt:
    '''
    gradient free optimizer
    We want to use it for get the action distribution
    QUESTION: HOW IS RELATED TO THE ORIGINAL STATE?
    TEMPORAL ANSWER: is not, for now is just the best distrib for a task specific
    '''

    def __init__(self, cost_funct,
                 population_shape,   # action dim * task horizon
                 numb_population: int = 10,
                 numb_elite: int = 5):
        self.numb_elite = numb_elite
        self.numb_population = numb_population
        self.population_shape = population_shape
        self.mean_vect = np.zeros(self.population_shape)
        self.std = np.ones(self.population_shape)

        self.population = [
            self.mean_vect + np.random.rand(self.population_shape)*self.std
            for _ in range(self.numb_population)
        ]

    def update(self, rewards: np.array):
        elite_idxs = np.array(rewards).argsort()[-self.numb_elite:]
        elite_weights = [self.population[idx] for idx in elite_idxs]
        self.mean_vect = np.array(elite_weights).mean()
        self.std = np.array(elite_weights).std()

        self.population = [
            self.mean_vect + np.random.rand(self.population_shape)*self.std
            for _ in range(self.numb_population)
        ]
    @property
    def solution(self):
        return self.mean_vect

    def sample_action(self):
        return self.population
