from config import *
import numpy as np

class ProspectAgent:
    def __init__(self, shape):
        self.q = np.random.uniform(-0.01, 0.01, shape + (ACTION_COUNT,)).astype(np.float32)

    def val(self, r):
        return r ** PROSPECT_ALPHA if r >= 0 else -PROSPECT_LAMBDA * ((-r) ** PROSPECT_BETA)

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return int(np.random.randint(ACTION_COUNT))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        r = self.val(reward)
        cur = self.q[state + (action,)]
        tar = r if done else r + GAMMA * float(np.max(self.q[next_state]))
        self.q[state + (action,)] = cur + ALPHA * (tar - cur)