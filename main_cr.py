import numpy as np
import gym
from evaluators import GymEvaluator
from algorithms import CR_FM_NES

if __name__ == '__main__':

    rs = np.random.RandomState()
    env = gym.make('CartPole-v0')

    d = 4
    lamb = 8
    m_init = 2 * rs.rand(d, 1) - 1
    sig = 0.5
    v = rs.randn(d, 1) / d
    D = np.ones((d, 1))

    M = 5
    T = 200
    eval_func = GymEvaluator(env, M, T).evaluate

    def stop_condition(best_eval, g):
        if best_eval >= M * T:
            return 1
        elif g > 2000:
            return -1
        else:
            return 0

    alg = CR_FM_NES(d, lamb, m_init, sig, v, D)
    is_success, data = alg.run(eval_func, stop_condition)
    print('success:{}'.format(is_success))

