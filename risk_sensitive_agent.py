from config import *
import numpy as np

class RiskSensitiveAgent:
    def __init__(self, shape):
        self.u = np.ones(shape + (ACTION_COUNT,), dtype=np.float64)
        self.q = np.zeros(shape + (ACTION_COUNT,), dtype=np.float64)

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return int(np.random.randint(ACTION_COUNT))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        idx = state + (action,)
        if done:
            tar = np.exp(-ETA * reward)
        else:
            m = max(float(np.min(self.u[next_state])), 1e-12)
            tar = np.exp(-ETA * reward) * (m ** GAMMA)
        self.u[idx] = (1 - ALPHA) * self.u[idx] + ALPHA * tar
        self.u[idx] = max(self.u[idx], 1e-12)
        self.q[idx] = (-1.0 / ETA) * np.log(self.u[idx])