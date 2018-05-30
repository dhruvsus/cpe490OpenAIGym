import gym
import time


def main(max_angle, max_dist):
    env = gym.make('CartPole-v1')

    for episode in range(30):
        observation = env.reset()
        action = 0

        for step in range(100):
            env.render()
            #print("Observation: ep", episode, " step", step, " :", observation)

            if observation[2] > (1/8)*max_angle:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode", episode, " finished after", step, " timesteps")
                print(observation)
                break


if __name__ == '__main__':
    MAX_ANGLE = 0.20944
    MAX_DIST = 2.4
    main(MAX_ANGLE, MAX_DIST)
