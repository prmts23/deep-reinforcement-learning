import random
import gym

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.python.util import memory

env = gym.make("CartPole-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n

# SHOW EXAMPLE

# for episode in range(20):
#     observation = env.reset()
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action = random.choice([0, 1])
#         observation, reward, done, info = env.step(action)
#         score += reward
#         if done:
#             print("Episode:{} Score:{}".format(episode, score))
#             break
# env.close()

# CREATE MODEL


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))

    return model


model = build_model(states, actions)

# show summary of the model
# model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )

    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])

# Trainer

# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Testing ageny
# scores = dqn.test(env, nb_episodes=5, visualize=False)
# print(np.mean(scores.history["episode_reward"]))

# Save agency
# dqn.save_weights("agency/dqn_weights.h5f", overwrite=True)


dqn.load_weights("agency/dqn_weights.h5f")


_ = dqn.test(env, nb_episodes=20, visualize=True)
