
import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum

import random
from collections import deque
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from gym import spaces
from freqtrade.persistence import Trade
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data import history
from freqtrade.resolvers import StrategyResolver
import gym
import datetime
import mpu

import json

from tb_callbacks import SaveOnStepCallback
from gym.utils import seeding


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class CustomEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000, window_size=50, stake_amount=200, fee=0.01, stop_loss=-0.15, punish_holding_amount=0, pair='1INCH/BUSD'):

        # Settings
        self.dataframe = df
        self.indicator_dataframe = None
        self.window_size = window_size
        self.pair = pair

        # Wallet
        self.initial_balance = initial_balance
        self.stake_amount = stake_amount
        self.opened_trade = None
        self.trades = []

        # Logic
        self.fee = fee
        self.stop_loss = stop_loss
        assert self.stop_loss <= 0, "`stoploss` should be less or equal to 0"
        self.punish_holding_amount = punish_holding_amount
        assert (
            self.punish_holding_amount <= 0
        ), "`punish_holding_amount` should be less or equal to 0"

        # Reward
        self._reward = 0
        self.total_reward = 0

        # Env settings
        _, number_of_features = self.dataframe.shape
        self.shape = (self.window_size, number_of_features)

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )

        self.seed()

    # 1

    def reset(self, env_steps_size=0):

        # Reset the state of the environment to an initial state

        self.opened_trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0

        self._current_tick = self.window_size + 1
        self._end_tick = len(self.dataframe) - 1

        return self._get_observation()
    # 2

    def _get_observation(self):

        return self.dataframe[(self._current_tick - self.window_size): self._current_tick].to_numpy()

    def _take_action(self, action):
        print("=== _take_action")

        if action == Actions.Hold.value:
            self._reward = self.punish_holding_amount
            if self.opened_trade != None:
                profit_percent = self.opened_trade.calc_profit_ratio(
                    rate=self.prices.loc[self._current_tick].open
                )
                if profit_percent <= self.stop_loss:
                    self._reward = profit_percent
                    self.opened_trade = None
            return

        if action == Actions.Buy.value:
            if self.opened_trade == None:
                self.opened_trade = Trade(
                    pair=self.pair,
                    open_rate=self.prices.loc[self._current_tick].open,
                    open_date=self.prices.loc[self._current_tick].date,
                    stake_amount=self.stake_amount,
                    amount=self.stake_amount /
                    self.prices.loc[self._current_tick].open,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                self.trades.append(
                    {
                        "step": self._current_tick,
                        "type": "buy",
                        "total": self.prices.loc[self._current_tick].open,
                    }
                )
            return

        if action == Actions.Sell.value:
            if self.opened_trade != None:
                profit_percent = self.opened_trade.calc_profit_ratio(
                    rate=self.prices.loc[self._current_tick].open
                )
                self.opened_trade = None
                self._reward = profit_percent

                self.trades.append(
                    {
                        "step": self._current_tick,
                        "type": "sell",
                        "total": self.prices.loc[self._current_tick].open,
                    }
                )
            return
    # Execute one time step within the environment

    def step(self, action):
        print("=== step")
       # Execute one time step within the environment
        done = False

        self._reward = 0

        if self._current_tick >= self._end_tick:
            done = True

        self._take_action(action)

        self._current_tick += 1

        self.total_reward += self._reward

        observation = self._get_observation()

        return observation, self._reward, done, {}


def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


"""Settings"""
PAIR = "1INCH/BUSD"
TRAINING_RANGE = "20210601-20210901"
WINDOW_SIZE = 50
LOAD_PREPROCESSED_DATA = False  # useful if you have to calculate a lot of features
SAVE_PREPROCESSED_DATA = True
LEARNING_TIME_STEPS = int(10000)
LOG_DIR = "./logs/"
TENSORBOARD_LOG = "./tensorboard/"
MODEL_DIR = "./models/"
"""End of settings"""
freqtrade_config = Configuration.from_files(["config.json"])
_preprocessed_data_file = "preprocessed_data.pickle"

strategy = StrategyResolver.load_strategy(freqtrade_config)

# Import datasets


def load_data(config, pair, timeframe, timerange, window_size):
    timerange = TimeRange.parse_timerange(timerange)

    return history.load_data(
        datadir=config["datadir"],
        pairs=[pair],
        timeframe=timeframe,
        timerange=timerange,
        startup_candles=window_size + 1,
        fail_without_data=True,
        data_format=config.get("dataformat_ohlcv", "json"),
    )


def get_dataframe():
    timeframe = freqtrade_config.get("timeframe")
    required_startup = strategy.startup_candle_count

    data = load_data(freqtrade_config, PAIR, timeframe,
                     TRAINING_RANGE, WINDOW_SIZE)
    if SAVE_PREPROCESSED_DATA:
        mpu.io.write(_preprocessed_data_file, data)

    pair_data = data[PAIR][required_startup:].copy()

    return pair_data


def main():
    df = get_dataframe()

    train_df = df[:-720-WINDOW_SIZE]
    test_df = df[-720-WINDOW_SIZE:]  # 30 days

    train_env = CustomEnv(train_df, window_size=WINDOW_SIZE)
    # test_env = CustomEnv(test_df, window_size=WINDOW_SIZE)

    train_env = Monitor(train_env, LOG_DIR)

    # Optional policy_kwargs
    # custom-network-architecture
    # see https: // stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight = policy_kwargs
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                      net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[
    #                      32, dict(pi=[64,  64], vf=[64, 64])])
    policy_kwargs = dict(net_arch=[32, dict(pi=[64, 64], vf=[64, 64])])

    model = PPO(  # See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html for other algos with discrete action space
        "MlpPolicy",
        train_env,
        verbose=0,
        device="auto",
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs=policy_kwargs,
    )
    start_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"{strategy.get_strategy_name()}_{train_env.env.__class__.__name__}_{model.__class__.__name__}_{start_date}"

    tb_callback = SaveOnStepCallback(
        check_freq=5000,
        save_name=f"best_model_{base_name}",
        save_dir=MODEL_DIR,
        log_dir=LOG_DIR,
        verbose=1,
    )

    print(
        f"You can run tensorboard with: 'tensorboard --logdir {Path(TENSORBOARD_LOG).absolute()}'"
    )
    print("Learning started.")

    model.learn(total_timesteps=LEARNING_TIME_STEPS, callback=tb_callback)
    model.save(f"{MODEL_DIR}final_model_{base_name}")

    # Random_games(train_env, train_episodes=100, training_batch_size=500)


# def Random_games(env, train_episodes=50, training_batch_size=500):
#     average_net_worth = 0
#     for episode in range(train_episodes):

#         state = env.reset(env_steps_size=training_batch_size)

#         while True:
#             env.render()

#             action = np.random.randint(3, size=1)[0]

#             state, reward, done = env.step(action)

#             if env.current_step == env.end_step:
#                 average_net_worth += env.net_worth
#                 print("net_worth:{} ".format(env.net_worth))
#                 break

#     print("average_net_worth:", average_net_worth/train_episodes)


if __name__ == "__main__":
    main()
