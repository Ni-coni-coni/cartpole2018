import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    M = 5
    T = 200
    alpha = 0.1