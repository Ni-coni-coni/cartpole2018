import numpy as np
import gym
from evaluators import GymEvaluator
from algorithms import HillClimbing

if __name__ == '__main__':

    rs = np.random.RandomState()
    env = gym.make('CartPole-v0')

    d = 4
    x_init = 2 * rs.rand(d, 1) - 1
    alpha = 0.1

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

    alg = HillClimbing(d, x_init, alpha)
    is_success, data = alg.run(eval_func, stop_condition)
    print('success:{}'.format(is_success))

