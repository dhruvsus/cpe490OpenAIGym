import gym
import numpy as np
from matplotlib import pyplot as plt


def main():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 200  # Cap CartPole to this amount
    num_samples = 5  # Number of times to utilize each algorithm

    num_episodes = []
    for i in range(num_samples):
        res = climb_hill(env)
        if res:
            num_episodes.append(res)

    plt.hist(num_episodes)
    plt.title('Hill Climbing Algorithm')
    plt.xlabel('Episodes')
    plt.ylabel('Frequency')
    plt.show()

    num_episodes = []
    for i in range(num_samples):
        num_episodes.append(run_guesses(env))

    plt.hist(num_episodes)
    plt.title('Random Guessing Algorithm')
    plt.xlabel('Episodes')
    plt.ylabel('Frequency')
    plt.show()


# Function used to test parameter set
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
        params = np.random.rand(4) * 2 - 1  # Random weights btwn -1 and 1
        reward = test_params(env, params)
        if reward > best_reward:
            best_reward = reward
            best_params = params
            t_found = t+1
            if reward == env._max_episode_steps:
                break

    display_results(env, best_params, best_reward, t_found)

    return t_found


def climb_hill(env, scale_noise=0.1):
    best_params = None
    best_reward = 0
    t_found = None

    # Start with random setting of the parameters
    params = np.random.rand(4) * 2 - 1
    print("Starting params of hill climb:", params)

    for t in range(10000):  # Number of iterations to "climb"
        noise = (np.random.rand(4) * 2 - 1) * scale_noise
        params = params + noise  # Add noise matrix to params
        reward = test_params(env, params)
        if reward > best_reward:
            best_reward = reward
            best_params = params
            t_found = t+1
            if reward == env._max_episode_steps:
                break

    display_results(env, best_params, best_reward, t_found)

    if best_reward == env._max_episode_steps:
        return t_found
    else:
        return None  # return 0 if we did not find desired reward

def display_results(env, params, reward, timesteps):
    print("Winning params:", params)
    print("Reward:", reward)
    print("Found after {} episodes".format(timesteps))

if __name__ == "__main__":
    main()
