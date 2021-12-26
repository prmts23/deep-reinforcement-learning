import ta

import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.agents import DQNAgent


cdd = CryptoDataDownload()

bitfinex_data = pd.concat([
    cdd.fetch("Bitfinex", "USD", "BTC", "1h").add_prefix("BTC:"),
    ], axis=1)


bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(list(bitfinex_data['BTC:close']), dtype="float").rename("USD-BTC")
)

bitfinex_btc = bitfinex_data.loc[:, [name.startswith("BTC") for name in bitfinex_data.columns]]

ta.add_all_ta_features(
    bitfinex_data,
    colprefix="BTC:",
    **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
)

with NameSpace("bitfinex"):
    bitfinex_streams = [
        Stream.source(list(bitfinex_btc[c]), dtype="float").rename(c) for c in bitfinex_btc.columns
    ]

feed = DataFeed(bitfinex_streams)


portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 10 * BTC),
])

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    window_size=15,
    enable_logger=False
)


done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

portfolio.ledger.as_frame().head(7)