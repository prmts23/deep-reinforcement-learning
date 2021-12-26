from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.exchanges.live import CCXTExchange

import ccxt
import pandas as pd


# GET BINANCE EXCHANGE LIVE USE

# binance = ccxt.binance()
# exchange = CCXTExchange(exchange=binance, base_instrument='USD')



import json

data = json.load(open("1INCH_BUSD-5m.json", "r"))

df = pd.DataFrame.from_dict(data, orient="columns")

df.rename(
    columns={0: "Date", 1: "Open", 2: "High", 3: "Low", 4: "Close", 5: "Volume"},
    inplace=True,
)

df["Date"] = pd.to_datetime(df["Date"], unit="ms")

df.set_index("Date", inplace=True)

print(df.dtypes)
print(df.head())


# environment = TradingEnvironment(exchange=exchange,
#                                  action_scheme=action_scheme,
#                                  reward_scheme=reward_scheme,
#                                  feature_pipeline=feature_pipeline)