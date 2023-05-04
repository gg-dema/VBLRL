import random
from collections import deque, defaultdict


class MultiEnvReplayBuffer:
    def __init__(self, buffer_size_per_env):
        self.buffer_size_per_env = buffer_size_per_env
        self.buffer = defaultdict(lambda: deque(maxlen=buffer_size_per_env))

    def add(self, state, action, reward, next_state, done, env_id):
        transition = (state, action, reward, next_state, done, env_id)
        self.buffer[env_id].append(transition)

    def sample_env(self, env_id, batch_size):
        transitions = random.sample(self.buffer[env_id], batch_size)
        return self._transpose_transitions(transitions)

    def sample_all_envs(self, batch_size):
        env_ids = list(self.buffer.keys())
        env_id = random.choice(env_ids)
        transitions = random.sample(self.buffer[env_id], batch_size)
        return self._transpose_transitions(transitions)

    def _transpose_transitions(self, transitions):
        states, actions, rewards, next_states, dones, env_ids = [], [], [], [], [], []
        for transition in transitions:
            state, action, reward, next_state, done, env_id = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            env_ids.append(env_id)
        return states, actions, rewards, next_states, dones, env_ids


    def write_buffer(self, path):
        pass

    def read_buffer(self, path):
        pass
