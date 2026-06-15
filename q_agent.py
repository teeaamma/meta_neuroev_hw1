from config import *
import numpy as np

class QLearningAgent:
    def __init__(self, shape):
        self.q = np.zeros(shape + (ACTION_COUNT,), dtype=np.float32)

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return int(np.random.randint(ACTION_COUNT))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        cur = self.q[state + (action,)]
        tar = reward if done else reward + GAMMA * float(np.max(self.q[next_state]))
        self.q[state + (action,)] = cur + ALPHA * (tar - cur)