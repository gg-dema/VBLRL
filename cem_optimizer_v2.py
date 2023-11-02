""" version 2 of cem optimizer, convert some stuff to numpy"""
import numpy as np


class CEM_opt:
    def __init__(
        self,
        num_action_seq=500,
        action_seq_len=4 * 20,  # 4 AKA act dim, 20 AKA horizon
        percent_elite=0.1,
    ):
        self.num_action_seq = 500
        self.action_seq_len = action_seq_len
        self.population = np.empty((num_action_seq, action_seq_len))
        self.percent_elite = percent_elite
        self.num_elite = int(num_action_seq * percent_elite)
        self.mean_vect = np.zeros(self.action_seq_len)
        self.std_vect = np.ones(self.action_seq_len) * 0.5
        self.min = -1
        self.max = +1
        self.interpolation_coef = 0.0
        self._generate_population()

    def _generate_population(self):
        for i in range(self.population.shape[0]):
            self.population[i] = self._generate_one_action_seq()

    def _generate_one_action_seq(self):
        return np.clip(
            self.mean_vect + np.random.randn(self.action_seq_len) * self.std_vect,
            self.min,
            self.max,
        )

    def update(self, rewards: np.array):
        elite_idx = rewards.argsort()[-self.num_elite :]
        elite_weights = self.population[elite_idx]
        self.mean_vect = self.mean_vect * self.interpolation_coef + (
            1 - self.interpolation_coef
        ) * elite_weights.mean(axis=0)
        self.std_vect = self.std_vect * self.interpolation_coef + (
            1 - self.interpolation_coef
        ) * elite_weights.std(axis=0)
        self.std_vect = np.clip(self.std_vect, 0.5, 1)
        self._generate_population()

    def solutions(self):
        return self.mean_vect
