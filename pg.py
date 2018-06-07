import gym
import numpy as np
from keras.models import Model
from keras import layers
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from matplotlib import pyplot as plt


def build_network(num_inputs, num_outputs):
    # use Keras functional API to construct policy network
    network = layers.Input(shape=(num_inputs,))
    input = network

    for layer in [32, 16, 8]:
        network = layers.Dense(layer)(network)
        network = layers.Activation("relu")(network)

    network = layers.Dense(num_outputs)(network)
    network = layers.Activation("softmax")(network)

    return Model(inputs=input, outputs=network)


def build_train(model, num_outputs):
    prob_action = model.output
    encoded_action = K.placeholder(shape=(None, num_outputs), name="encoded_action")
    discount_reward = K.placeholder(shape=(None,), name="discount_reward")

    loss = K.mean(-K.log(K.sum(prob_action * encoded_action, axis=1)) * discount_reward)

    deltas = optimizers.Adam().get_updates(params=model.trainable_weights, loss=loss)

    return K.function(inputs=[model.input, encoded_action, discount_reward], outputs=[], updates=deltas)


def get_action(model, state, num_outputs):
    state = np.expand_dims(state, axis=0)
    prob_action = np.squeeze(model.predict(state))

    return np.random.choice(np.arange(num_outputs), p=prob_action)


def fit(train, states, actions, rewards, num_outputs):
    encoded_action = np_utils.to_categorical(actions, num_classes=num_outputs)
    discount_reward = get_discounted(rewards)
    train([states, encoded_action, discount_reward])


def get_discounted(rewards):
    discount_rate = 0.96
    discounted = np.zeros_like(rewards, dtype=np.float32)
    sum = 0

    for t in reversed(range(len(rewards))):
        sum = sum * discount_rate + rewards[t]
        discounted[t] = sum 

    discounted -= discounted.mean() / discounted.std()

    return discounted


def run_episode(env, train, model, num_outputs):
    is_done = False
    state = env.reset()
    states = []
    actions = []
    rewards = []
    total_reward = 0

    while not is_done:
        action = get_action(model, state, num_outputs)
        new_state, reward, is_done, info = env.step(action)
        total_reward += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state

    fit(train, np.asarray(states), np.asarray(actions), np.asarray(rewards), num_outputs)

    return total_reward


def reset_model(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def main():
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 200
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    model = build_network(num_inputs, num_outputs)
    train = build_train(model, num_outputs)

    steps_per_sample = []
    for sample in range(10):
        for episode in range(5000):
            reward = run_episode(env, train, model, num_outputs)
            if reward == env._max_episode_steps:
                print("Found reward {} after {} episodes".format(reward, episode))
                steps_per_sample.append(episode)
                break
        reset_model(model)

    plt.hist(steps_per_sample)
    plt.title('Policy Gradient Algorithm')
    plt.xlabel('Episodes')
    plt.ylabel('Frequency')
    plt.show()
     


    env.close()


if __name__ == '__main__':
    main()
