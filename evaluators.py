import numpy as np


class GymEvaluator(object):
    def __init__(self, env, M=5, T=200):
        self.env = env
        self.M = M
        self.T = T

    def set_seed(self, seed):
        self.env.seed(seed)

    def evaluate(self, x):
        r_sum = 0
        for episode in range(self.M):
            state = self.env.reset()
            for t in range(self.T):
                action = 0 if np.vdot(x, state) > 0 else 1
                state, reward, done, _ = self.env.step(action)
                r_sum += reward
                if done:
                    break
        return r_sum
