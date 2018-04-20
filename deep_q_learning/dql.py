#!/usr/bin/env python

import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import deque

import gym

env = gym.make("CartPole-v0")

# --==:: Network Model ::==--
# Input: two consecutive game states
# Output: Q-values of actions

model = Sequential()
model.add(Dense(20, input_shape=(2, 4), activation="relu", kernel_initializer="normal"))
model.add(Flatten())
model.add(Dense(18, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="relu", kernel_initializer="normal"))
model.add(Dense(env.action_space.n, activation="linear", kernel_initializer="normal"))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

D = deque()  # Register where the actions will be stored
inputs_shape = (0, 0)
mb_size = 500


def observe(observe_time=50000, epsilon=0.7):
    observation = env.reset()

    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)

    global inputs_shape
    inputs_shape = (mb_size,) + state.shape[1:]

    for t in range(observe_time):

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)
            action = np.argmax(Q)

        observation_new, reward, done, info = env.step(action)
        obs_new = np.expand_dims(observation_new, axis=0)

        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)

        D.append((state, action, reward, state_new, done))

        state = state_new
        if done:
            env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)


def learn(gamma=0.99):
    mini_batch = random.sample(D, mb_size)  # Sample some moves
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(mb_size):
        state, action, reward, state_new, done = mini_batch[i]
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

        model.train_on_batch(inputs, targets)  # Train network to output the Q function


def play(render=False):
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        if render:
            env.render()
        Q = model.predict(state)
        action = np.argmax(Q)
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward
    return tot_reward


def main():
    num_epochs = 10
    for epoch in range(num_epochs):
        print("Learning round: %d" % epoch)
        epsilon = max(0, 0.5 - epoch / num_epochs)
        observe(epsilon=epsilofn)
        learn()
        avg_reward = np.mean([play() for _ in range(10)])
        print("Average reward: %s" % avg_reward)
    print("Finished Learning")

    print("Playing...")
    reward = play(render=True)
    print('Game ended! Total reward: %s' % reward)


if __name__ == "__main__":
    main()
