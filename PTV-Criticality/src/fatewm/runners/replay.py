import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, k=batch_size)
        obs, act, rew, nxt, done = zip(*batch)
        return (np.stack(obs), np.array(act), np.array(rew, dtype=np.float32),
                np.stack(nxt), np.array(done, dtype=np.float32))
