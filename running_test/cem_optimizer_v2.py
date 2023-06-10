import numpy as np
import scipy.stats as stats

class CEM_opt:
    '''
    gradient free optimizer
    We want to use it for get the action distribution
    QUESTION: HOW IS RELATED TO THE ORIGINAL STATE?
    TEMPORAL ANSWER: is not, for now is just the best distrib for a task specific
    '''
    def __init__(self,
                 population_shape,   # (action dim, task horizon)
                 num_population: int,
                 num_elite: int = 30):
        self.num_elite = num_elite
        self.num_population = num_population
        self.population_shape = population_shape
        self.distribution = stats.truncnorm(-2, 2,
                                            loc=np.zeros((self.population_shape, self.num_population)),
                                            scale=np.ones((self.population_shape, self.num_population)),
                                            )
        self.mean = np.zeros((self.population_shape, self.num_population))
        self.var = np.ones((self.population_shape, self.num_population))*0.5
        self.population = self.generate_pop()

    def generate_pop(self):
        rand = self.distribution.rvs(size=[self.population_shape, self.num_population])
        return rand*np.sqrt(self.var) + self.mean


    def update(self, rewards: list):
        rewards = np.array(rewards)
        elite = self.population[np.argsort(-rewards)[:self.num_elite]]
        #print('mean', elite_weights.mean(axis=0))
        #print('real', elite_weights)
        #print('std', elite_weights.std(axis=0))
        self.mean = np.mean(elite, axis=0)
        self.var = np.var(elite, axis=0)
        self.population = self.generate_pop()

    @property
    def solution(self):
        return self.population.mean(axis=0)

    def sample_act(self):
        return self.population



"""
def cem(vect, 
        n_iter=500, 
        max_timestep=1000, 
        pop_size=50, 
        elite=2, 
        std=0.5
        ):
    
    reward = []
    best_w = std*np.random.randn(vect.shape[0])
    
    for iter in range(n_iter):
        weight_pop = [best_w + (std*np.random.randn(vect.shape[0])) for i in range(pop_size)]
        rewards = np.array([eval(w, max_timestep) for w in weight_pop])
    """

