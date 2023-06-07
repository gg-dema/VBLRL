import torch
import os
import random
import pickle
import numpy as np
from collections import deque, defaultdict

class SingleTaskReplayBuffer:
    pass

class MultiEnvReplayBuffer:
    """
    Generic struct :
    dict[key for each env: int] =
    ---------- deque(store tuple) = each tuple = 1 transition
    ------------------------------- = (transition) = (state: np.array,
                                                      action: np.array,
                                                      rewards: np.float,
                                                      next_state: np.array,
                                                      dones: bool)
    --------------------------------------------------------------
    for save a local copy of the buffer :
      automatically save each deque, in several file.
      1 file for env

    """
    def __init__(self, buffer_size_per_env, **kwargs):
        self.size_per_env = buffer_size_per_env
        self.buffer_size_per_env = buffer_size_per_env
        self._initialize_empty_buffer()
        if kwargs.get('preload'):  # get handle keyError, basic dict[key] dont
            self.read_buffer(kwargs['path_preload'])

    def add(self, state, action, reward, next_state, done, env_id):
        transition = (state, action, reward, next_state, done)
        self.buffer[env_id].append(transition)


    def sample_env(self, env_id, batch_size):
        transitions = random.sample(self.buffer[env_id], batch_size)
        return self._transpose_transitions(transitions)

    def sample_all_envs(self, batch_size):
        # NON FUNZIONA ---> env_id e' un int, quindi seleziona transizioni solo da un buffer
        env_ids = list(self.buffer.keys())
        env_id = random.choice(env_ids)
        transitions = random.sample(self.buffer[env_id], batch_size)
        return self._transpose_transitions(transitions)

    def _transpose_transitions(self, transitions):
        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.float32)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32).reshape((-1, 1))
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones = np.array([t[4] for t in transitions])
        return states, actions, rewards, next_states, dones

    def _initialize_empty_buffer(self):
        self.buffer = defaultdict(lambda: deque(maxlen=self.buffer_size_per_env))

    def write_buffer(self, path):
        #convert default dict to basic dict : easier to save
        for env_id in self.buffer.keys():
          file_path = os.path.join(path, f'buffer_env{env_id}.pkl')
          with open(file_path, 'wb') as file:
            #sto salvando solo le deque
            pickle.dump(self.buffer[env_id], file)

    def read_buffer(self, path):
        self._initialize_empty_buffer()
        files = os.listdir(path)

        buff_key = []
        for file in files:
            if file.endswith('pkl'): buff_key.append(file[-5])

        for k in buff_key:
            buffer_path = os.path.join(path, f'buffer_env{k}.pkl')
            with open(buffer_path, 'rb') as local_buff:
                temp_buff = pickle.load(local_buff)
                self.buffer[int(k)] += temp_buff

