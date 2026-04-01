import random
from collections import deque
import numpy as np

class SeqReplayBuffer:
    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buf = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, next_obs, done, extra=None):
        """Store a transition.

        extra is an optional dict for method-specific supervision signals,
        e.g., oracle candidate sets and costs for decision-aligned energy learning.
        """
        self.buf.append((obs, action, reward, next_obs, done, extra))
        
    def __len__(self):
        return len(self.buf)
        
    def sample(self, batch_size: int):
        valid_starts = [i for i in range(len(self.buf) - self.seq_len + 1)]
        if not valid_starts:
            valid_starts = [0]
            
        starts = random.choices(valid_starts, k=batch_size)
        
        obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []
        extra_seq = []
        
        for st in starts:
            o_s, a_s, r_s, d_s = [], [], [], []
            e_s = []
            for i in range(self.seq_len):
                idx = min(st + i, len(self.buf) - 1)
                obs, act, rew, nxt, done, extra = self.buf[idx]
                o_s.append(obs)
                a_s.append(act)
                r_s.append(rew)
                d_s.append(done)
                e_s.append(extra)
            obs_seq.append(np.stack(o_s))
            act_seq.append(np.array(a_s))
            rew_seq.append(np.array(r_s, dtype=np.float32))
            done_seq.append(np.array(d_s, dtype=np.float32))
            extra_seq.append(e_s)
            
        return (np.stack(obs_seq), np.stack(act_seq), np.stack(rew_seq), np.stack(done_seq), extra_seq)
