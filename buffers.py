import os
import random
import pickle
import numpy as np
from collections import deque, defaultdict


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
        self.elem_for_buffer = defaultdict(lambda: 0)
        if kwargs.get('preload'):  # get handle keyError, basic dict[key] dont
            self.read_buffers(kwargs['path_preload'])

    def add(self, state, action, reward, next_state, done, env_id):

        transition = (state, action, reward, next_state, done)

        if self.elem_for_buffer[env_id] < self.buffer_size_per_env:
            self.elem_for_buffer[env_id] += 1

        self.buffers[env_id].append(transition)


    def sample_env(self, env_id, batch_size):
        transitions = random.sample(self.buffers[env_id], batch_size)
        return self._transpose_transitions(transitions)

    def sample_all_envs(self, batch_size):
        batch = [None]*batch_size
        max_id = max(self.buffers.keys())
        for i in range(batch_size):
            selected_buffer_idx = random.randint(0, max_id)
            elem_idx = random.randint(0, self.elem_for_buffer[selected_buffer_idx])
            batch[i] = self.buffers[selected_buffer_idx][elem_idx]

        return self._transpose_transitions(batch)


    def _transpose_transitions(self, transitions):

        states = np.array([t[0] for t in transitions], dtype=np.float64)
        actions = np.array([t[1] for t in transitions], dtype=np.float64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float64).reshape((-1, 1))
        next_states = np.array([t[3] for t in transitions], dtype=np.float64)
        dones = np.array([t[4] for t in transitions])
        return states, actions, rewards, next_states, dones

    def _initialize_empty_buffer(self):
        self.buffers = defaultdict(lambda: deque(maxlen=self.buffer_size_per_env))

    def write_buffer(self, path):
        #convert default dict to basic dict : easier to save
        for env_id in self.buffers.keys():
          file_path = os.path.join(path, f'buffer_env{env_id}.pkl')
          with open(file_path, 'wb') as file:
            #sto salvando solo le deque
            pickle.dump(self.buffers[env_id], file)

    def read_buffers(self, path, from_scratch=False):

        if from_scratch:
            self._initialize_empty_buffer()

        files = os.listdir(path)
        buff_key = []

        for file in files:
            if file.endswith('pkl'): buff_key.append(file[-5])

        for k in buff_key:
            buffer_path = os.path.join(path, f'buffer_env{k}.pkl')
            with open(buffer_path, 'rb') as local_buff:
                temp_buff = pickle.load(local_buff)
                self.buffers[int(k)] += temp_buff
                self.elem_for_buffer[int(k)] = len(self.buffers[int(k)])


