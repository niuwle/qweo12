from client import binance_client
import logging
import numpy as np
import talib
import math

def get_min_notional(symbol):
    exchange_info = binance_client.exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    return float(f['minNotional'])
    return None

def print_wallet_balances():
    # Get account information
    account_info = binance_client.account()

    # Get balances
    balances = account_info['balances']

    # Print each coin and its balance
    for coin in balances:
        logging.info(f"Coin: {coin['asset']}, Free: {coin['free']}, Locked: {coin['locked']}")

def calculate_success_rate(buy_score, sell_score):
    total_score = buy_score + sell_score
    return buy_score / total_score if total_score else 0

def calculate_supertrend(high_prices, low_prices, close_prices, period, multiplier):
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
    hl2 = (high_prices + low_prices) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    supertrend = np.zeros(len(close_prices))

    for i in range(period, len(close_prices)):
        if close_prices[i] <= upper_band[i]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i - 1]

        if close_prices[i] >= lower_band[i]:
            supertrend[i] = lower_band[i]
        else:
            if supertrend[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i - 1]

    return supertrend

def calculate_fibonacci_retracement_levels(high, low):
    diff = high - low
    level1 = high - 0.236 * diff
    level2 = high - 0.382 * diff
    level3 = high - 0.618 * diff
    return [level1, level2, level3]

def calculate_pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def round_step_size(quantity, step_size):
    precision = int(round(-math.log(step_size, 10), 0))
    return round(quantity, precision)
