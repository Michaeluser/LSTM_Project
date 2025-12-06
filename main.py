import datetime
import os
import matplotlib.pyplot as plt
import sys

from agents.agent import Agent, AgentConfig
from loaders.pred_prices_loader_csv import PredPricesLoaderCSV2
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_loader_csv import RealPricesLoaderCSV
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval


def utc_datetime(year: int, month: int, day: int):
    return datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)

if __name__ == '__main__':
    # AGENT_SYMBOLS = [                 #better to refer to them as features since symbols is harder to understand, maybe cryptocurrencies
    #     'BINANCE_SPOT_DOT_USDT',
    #     'BINANCE_SPOT_SOL_USDT',
    #     'BINANCE_SPOT_BNB_USDT',
    #     'BINANCE_SPOT_LINK_USDT',
    #     'BINANCE_SPOT_ADA_USDT',
    #     'BINANCE_SPOT_AVAX_USDT',
    #     'BINANCE_SPOT_LTC_USDT',
    #     'BINANCE_SPOT_XRP_USDT',
    #     'BINANCE_SPOT_BTC_USDT',
    #     'BINANCE_SPOT_ICP_USDT',
    #     'BINANCE_SPOT_TRX_USDT',
    #     'BINANCE_SPOT_SHIB_USDT',
    #     'BINANCE_SPOT_ETH_USDT',
    #     'BINANCE_SPOT_DOGE_USDT',
    # ]

    AGENT_SYMBOLS = [
        'BINANCE_SPOT_DOT_USDT',
    ]

    ACTION_LONG = 0
    ACTION_SHORT = 1
    MODEL_ACTIONS = [ACTION_LONG, ACTION_SHORT]

    AGENT_PRED_CSV_FILE = os.getcwd() + '/data/predictions.csv'
    AGENT_REAL_DAILY_PRICES_PROVIDER = RealPricesProvider(
        real_prices_loader=RealPricesLoaderCSV(
            prices_csv_file=os.getcwd() + '/data/prices_updated.csv'
        ),
        filter_symbols=AGENT_SYMBOLS
    )

    AGENT_PRED_MODELS = [
        'moirai_base',
        'moirai_large',
        'chronos',
        'tirex',
        'sundial'
    ]

    AGENT_PRED_PRICES_PROVIDER = PredPricesProvider(
        predicted_prices_loader=PredPricesLoaderCSV2(
            prices_file=AGENT_PRED_CSV_FILE,
            filter_models=AGENT_PRED_MODELS,
        )
    )

    AGENT_TRAIN_INTERVAL = TimeseriesInterval(
        date_from=utc_datetime(2023, 10, 1),
        date_to=utc_datetime(2025, 1, 1),
        time_unit='D'
    )

    AGENT_TEST_INTERVAL = TimeseriesInterval(
        date_from=utc_datetime(2025, 1, 1),
        date_to=utc_datetime(2025, 10, 31),
        time_unit='D'
    )

    config = AgentConfig()

    agent = Agent(
        config=config,
        real_prices_provider=AGENT_REAL_DAILY_PRICES_PROVIDER,
        pred_prices_provider=AGENT_PRED_PRICES_PROVIDER,
        symbols_amnt = len(AGENT_SYMBOLS) + len(AGENT_PRED_MODELS) * 3
    )

    for symbol in AGENT_SYMBOLS:
        print(symbol)
        error_rtt_train, batch_rtt = agent.train(interval = AGENT_TRAIN_INTERVAL, symbols = [symbol])
        error_rtt_test, freq_rate_test = agent.test(interval = AGENT_TEST_INTERVAL, symbols = [symbol])


    
    timeline_test = [ts * freq_rate_test for ts in range(len(error_rtt_test))]
    timeline_train = [ts for ts in batch_rtt]

    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(timeline_train, error_rtt_train)
    plt.title('Training Loss')
    plt.grid()
    # fig1.show()  # not needed if you call plt.show() later

    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(timeline_test, error_rtt_test, color='red')
    plt.title('Test Loss')
    plt.grid()

    plt.show()