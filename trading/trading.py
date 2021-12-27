# from datetime import datetime, timezone

# from pandas.core import frame
# import gym

# import gym_anytrading

# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import A2C

# import numpy as np
# import pandas as pd

# from matplotlib import pyplot as plt


import json


data = json.load(open("1INCH_BUSD-5m.json", "r"))

df = pd.DataFrame.from_dict(data, orient="columns")

df.rename(
    columns={0: "Date", 1: "Open", 2: "High", 3: "Low", 4: "Close", 5: "Volume"},
    inplace=True,
)


df["Date"] = pd.to_datetime(df["Date"], unit="ms")

df.set_index("Date", inplace=True)

# print(df.dtypes)
# print(df.head())


# Init
# env = gym.make('stocks-v0', df=df, frame_bound=(5,200), window_size=5)
# print(env.prices)
# print(env.signal_features)
# state = env.reset()

# while True:
#     action = env.action_space.sample()
#     n_state, reward, done, info = env.step(action)
#     if done:
#         print(info)
#         break

# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()


# env_maker = lambda: gym.make("stocks-v0", df=df, frame_bound=(15, 500), window_size=15)
# env = DummyVecEnv([env_maker])

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000)

# model.save("1inch_model")

# del model  # remove to demonstrate saving and loading


model = A2C.load("1inch_model")


env = gym.make("stocks-v0", df=df, frame_bound=(15, 500), window_size=15)
state = env.reset()

while True:
    obs = state[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        print(info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()
