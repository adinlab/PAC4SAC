import numpy as np
import os, torch, copy

class ExperienceMemory:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self,args):
        self.device = "cpu"
        self.buffer_size = 1000000
        self.field_names = ["state", "action", "reward", "next_state", "terminated", "step"]
        self.dims = args.dims
        self.reset()

    def reset(self, buffer_size=None):
        if buffer_size is not None:
            self.buffer_size = buffer_size
        self.data_size = 0
        self.pointer = 0
        self.memory = {field: np.empty(self.dims[field]) for field in self.field_names}

    def add(self, state, action, reward, next_state, terminated, step):
        for field, value in zip(self.field_names, [state, action, reward, next_state, terminated, step]):
            self.memory[field][self.pointer] = value
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.data_size = min(self.data_size + 1, self.buffer_size)

    def sample_by_index(self, index):
        return tuple(torch.from_numpy(self.memory[field][index]).to(self.device).float() for field in self.field_names)

    def sample_by_index_fields(self, index, fields):
        if len(fields) == 1:
            return torch.from_numpy(self.memory[fields[0]][index]).to(self.device).float() # return a tensor
        return tuple(torch.from_numpy(self.memory[field][index]).to(self.device).float() for field in fields)

    def sample_random(self, batch_size):
        index = np.random.choice(self.data_size, batch_size)
        return self.sample_by_index(index)

    def filter_by_nonterminal_steps_with_horizon(self, horizon):
        all_indices = np.arange(self.data_size - horizon + 1)
        terminal_indices = np.argwhere(self.memory["terminated"] == True)
        if terminal_indices.size == 0:
            return all_indices

        terminal_with_horizon_indices = np.array([np.arange(terminal - horizon+2, terminal + 1)for terminal in terminal_indices]).flatten()
        nonterminal_indices = np.setdiff1d(all_indices, terminal_with_horizon_indices)
        return nonterminal_indices

    def sample_random_sequence_snippet(self, batch_size, sequence_length):
        non_terminal_indices = self.filter_by_nonterminal_steps_with_horizon(sequence_length)
        indices = np.random.choice(non_terminal_indices, size=batch_size)
        output = []
        for i in range(sequence_length):
            output.append(self.sample_by_index(indices + i))
        return output

    def sample_all(self):
        return self.sample_by_index(range(self.data_size))

    def clone(self, other_memory):
        self.data_size = other_memory.data_size
        self.memory = copy.deepcopy(other_memory.memory)

    def extend(self, other_memory):
        for field in self.field_names:
            self.memory[field].extend(other_memory.memory[field])
        self.data_size = len(self.memory[field])

    def __len__(self):
        return self.data_size

    @property
    def size(self):
        return self.data_size

    def save(self, path):
        np.save(os.path.join(path, "experience_memory.npy"), self.memory)

    def get_last_observation(self):
        return self.sample_by_index([-1])

    def get_last_observations(self, batch_size):
        return self.sample_by_index(range(-batch_size, 0))