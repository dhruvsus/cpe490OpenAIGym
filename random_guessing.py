import gym
import numpy as np
from matplotlib import pyplot as plt


def main():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500  # Cap CartPole to this amount
    num_episodes = []

    # Do the random guessing algorithm N amount of times, record time it takes
    # to reach desired reward (env._max_episodes_steps)
    for i in range(1000): # Values in frequency histogram
        num_episodes.append(run_guesses(env))

    #print(num_episodes)
    plt.hist(num_episodes)
    plt.xlabel('Num Episodes')
    plt.ylabel('Frequency')
    plt.show()


def test_params(env, params):
    observation = env.reset()
    total_reward = 0
    while True:
        #env.render()
        res = np.matmul(params, observation)
        if res < 0:
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        #print(observation)
        total_reward += reward
        if done:
            break
    return total_reward


def run_guesses(env):
    best_params = None
    best_reward = 0
    t_found = None

    for t in range(10000):  # 10,000 random configurations, 1 each episode
        params = np.random.rand(4) * 2 - 1 # Random weights btwn -1 and 1
        reward = test_params(env, params)
        if reward > best_reward:
            #print("Params: {} @ {} episodes; Reward: {}".format(params, t+1, reward))
            best_reward = reward
            best_params = params
            t_found = t+1
            if reward == env._max_episode_steps:
                break

    print("Winning params:", best_params)
    print("Reward:", best_reward)
    print("Found after {} episodes".format(t_found))

    return t_found


if __name__ == "__main__":
    main()
