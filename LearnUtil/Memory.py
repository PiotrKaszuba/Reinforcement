import random
from collections import deque

import numpy as np


class Memory:
    chance_factor = 1.5e-7

    def __init__(self, length, previous_states=0, save_and_reset=False, model_saving=None):
        self._memory = deque(maxlen=length+previous_states)
        self._length = length
        self._previous_states = previous_states
        self._save_and_reset = save_and_reset
        assert not (save_and_reset and model_saving is None)
        self._model_saving = model_saving

    def store(self, remembrance, i_episode=None):
        self._memory.appendleft(remembrance)
        if self._save_and_reset and self.free_memory() == 0:
            if i_episode is None:
                i_episode = -np.random.randint(1, 10000000)
            self._model_saving.save_something(self._memory, '_MEMORIES_' + str(i_episode))
            self._memory.clear()

    def remember(self, state, action, reward, new_state, done, i_episode=None):
        self.store({'state': state, 'action': action, 'new_state': new_state, 'reward': reward, 'done': done},
                   i_episode)

    def rememberGymObservation(self, observation, i_episode=None):
        self.remember(observation[0], observation[1], observation[2], observation[3], observation[4], i_episode)

    def free_memory(self):
        return self._length - len(self._memory) + self._previous_states

    def get_memory(self):
        return self._memory

    def array_memory(self):
        return np.array(self._memory)[:self._length]

    def get_nth_newest(self, n=0):
        return self._memory[n]

    def memory_length(self):
        return len(self._memory)-self._previous_states

    def random_sample(self, n):
        sample = random.sample(range(0, len(self._memory) - self._previous_states), n)
        tempList = list(self._memory)
        x = [tempList[x:x + 1 + self._previous_states] for x in sample]
        return x

    def get_chain_states(self, sample):
        return [[y['state'] for y in x] for x in sample]

    def episode_batch_size_chance(self, i_episode, batch_size):
        return i_episode * self.chance_factor * batch_size

    def replace_first_with_specified_value_by_chance(self, batch, chance, value_type, value):
        if np.random.rand() < chance:
            specified_value = self.sample_specified_value(value_type, value)
            if specified_value is not None:
                batch.pop()
                batch.append(specified_value)
        return batch

    def sample_specified_value(self, value_type, value):
        index = next((ind for ind, x in enumerate(self._memory) if x[value_type] == value), None)
        if index is None or index >= len(self._memory) - self._previous_states:
            return None
        else:
            tempList = list(self._memory)
            return tempList[index:index + 1 + self._previous_states]

