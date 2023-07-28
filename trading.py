import numpy as np
import client
import logging
import utils
import talib
import config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from textblob import TextBlob
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

newsapi = config.newsapi

def process_coin(symbol, ticker_prices, balances):
    logging.info(f'Processing coin: {symbol}')
    logging.info(f'Ticker prices: {ticker_prices}')
    logging.info(f'Balances: {balances}')
    current_price = ticker_prices.get(symbol, None)
    if current_price is None:
        logging.error(f'No price data for {symbol}')
        return

    coin_balance = balances.get(symbol[:-4], 0)

    coin_balance_usdt = coin_balance * current_price

    logging.info(f'Current price of {symbol}: {current_price}')
    logging.info(f'Balance of {symbol}: {coin_balance} ({coin_balance_usdt} USDT)')

    ohlcv = client.client.klines(symbol, '1m')
    close_prices = [float(x[4]) for x in ohlcv]
    high_prices = [float(x[2]) for x in ohlcv]
    low_prices = [float(x[3]) for x in ohlcv]

    close_prices_train, close_prices_test, high_prices_train, high_prices_test, low_prices_train, low_prices_test = train_test_split(close_prices, high_prices, low_prices, test_size=0.2, shuffle=False)

    high_prices_train = np.array(high_prices_train)
    low_prices_train = np.array(low_prices_train)
    close_prices_train = np.array(close_prices_train)

    rsi = talib.RSI(close_prices_train, 14)
    macd, signal, hist = talib.MACD(close_prices_train, 12, 26)
    sma = talib.SMA(close_prices_train, 14)
    ema = talib.EMA(close_prices_train, 14)
    atr = talib.ATR(high_prices_train, low_prices_train, close_prices_train, 14)
    parabolic_sar = talib.SAR(high_prices_train, low_prices_train, 0.02, 0.2)
    upper_band, middle_band, lower_band = talib.BBANDS(close_prices_train, 20, 2, 2)
    supertrend = utils.calculate_supertrend(high_prices_train, low_prices_train, close_prices_train, 14, 3)

    fibonacci_levels = utils.calculate_fibonacci_retracement_levels(max(close_prices), min(close_prices))
    pivot, r1, s1, r2, s2 = utils.calculate_pivot_points(max(high_prices), min(low_prices), close_prices[-1])

    # Add more indicators
    stoch = talib.STOCH(high_prices_train, low_prices_train, close_prices_train, 5, 3, 0, 3, 0)
    cci = talib.CCI(high_prices_train, low_prices_train, close_prices_train, 14)
    mfi = talib.MFI(high_prices_train, low_prices_train, close_prices_train, np.array(volume), 14)

    # Fundamental analysis
    coin_info = client.client.get_coin_info(symbol[:-4])
    market_cap = coin_info['market_cap']
    total_supply = coin_info['total_supply']

    # Market sentiment
    news = newsapi.get_everything(q=symbol[:-4], from_param=datetime.now() - timedelta(days=1), to=datetime.now(), language='en', sort_by='relevancy')
    sentiment = TextBlob(news['articles'][0]['description']).sentiment.polarity

    # Machine learning model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(np.array([rsi, macd, signal, hist, sma, ema, atr, parabolic_sar, upper_band, middle_band, lower_band, supertrend, stoch, cci, mfi, market_cap, total_supply, sentiment]).T, close_prices_train)
    prediction = model.predict(np.array([rsi[-1], macd[-1], signal[-1], hist[-1], sma[-1], ema[-1], atr[-1], parabolic_sar[-1], upper_band[-1], middle_band[-1], lower_band[-1], supertrend[-1], stoch[-1], cci[-1], mfi[-1], market_cap, total_supply, sentiment]).reshape(1, -1))

    # Buy if the predicted price is higher than the current price
    if prediction > current_price:
        place_order(symbol, 'BUY', spending_limit, close_prices[-1])
    # Sell if the predicted price is lower than the current price
    elif prediction < current_price:
        place_order(symbol, 'SELL', spending_limit, close_prices[-1])


def rebalance_portfolio():
    logging.info(f'Rebalancing portfolio...')
    balance = client.client.account()
    total_balance = sum([float(coin['free']) * float(client.client.ticker_price(coin['asset'] + 'USDT')['price']) for coin in balance['balances'] if coin['asset'] != 'USDT'])
    target_investment = total_balance * config.investment_percentage

    for coin in balance['balances']:
        if coin['asset'] != 'USDT':
            coin_balance = float(coin['free']) * float(client.client.ticker_price(coin['asset'] + 'USDT')['price'])
            if coin_balance > target_investment:
                # Sell excess
                order = client.client.new_order(symbol=coin['asset'] + 'USDT', side='SELL', type='MARKET', quantity=(coin_balance - target_investment) / float(client.client.ticker_price(coin['asset'] + 'USDT')['price']))
            elif coin_balance < target_investment:
                # Buy deficit
                order = client.client.new_order(symbol=coin['asset'] + 'USDT', side='BUY', type='MARKET', quantity=(target_investment - coin_balance) / float(client.client.ticker_price(coin['asset'] + 'USDT')['price']))
def get_top_coins(limit):
    logging.info(f'Getting top {limit} coins...')
    exchange_info = client.client.exchange_info()
    symbols = [symbol_info['symbol'] for symbol_info in exchange_info['symbols'] if symbol_info['quoteAsset'] == 'USDT']
    tickers = [client.client.ticker_24hr(symbol) for symbol in symbols]
    
    # Calculate the price increase for each coin
    for ticker in tickers:
        ticker['price_increase'] = (float(ticker['lastPrice']) - float(ticker['openPrice'])) / float(ticker['openPrice'])
    
    # Filter out coins with low price increase
    tickers = [ticker for ticker in tickers if ticker['price_increase'] > 0.01]  # Adjust the threshold as needed
    
    # Sort the coins by their price increase
    coins = sorted(tickers, key=lambda x: x['price_increase'], reverse=True)
    
    return coins[:limit]  # Return the top 'limit' coins


def place_order(symbol, side, spending_limit, close_price):
    # Get the precision for this coin
    symbol_info = [s for s in client.client.exchange_info()['symbols'] if s['symbol'] == symbol][0]
    lot_size_info = [f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0]

    min_qty = float(lot_size_info['minQty'])
    max_qty = float(lot_size_info['maxQty'])
    step_size = float(lot_size_info['stepSize'])

    # Calculate the quantity
    quantity = spending_limit / close_price
    # Adjust the quantity to meet the LOT_SIZE filter requirements
    quantity = max(min_qty, min(max_qty, quantity))
    quantity = utils.round_step_size(quantity, step_size)

    # Adjust the quantity to meet the precision requirements
    precision = int(-np.log10(step_size))
    quantity = round(quantity, precision)

    # Ensure the quantity is within the minQty and maxQty limits
    quantity = max(min_qty, min(max_qty, quantity))

    # Get the minimum notional value for the trading pair
    min_notional = utils.get_min_notional(symbol)

    # Ensure the quantity meets the minimum notional value
    if quantity * close_price < min_notional:
        quantity = min_notional / close_price

    # Round the price to the correct precision
    close_price = round(close_price, precision)

    # Place a buy order
    order = client.client.new_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    # Set stop loss and take profit levels
    stop_loss = close_price * 0.9
    take_profit = close_price * 1.1
    # Place a stop loss order
    stop_loss_order = client.client.new_order(symbol=symbol, side='SELL', type='STOP_LOSS_LIMIT', quantity=spending_limit / close_price, price=stop_loss, stopPrice=stop_loss)
    # Place a take profit order
    take_profit_order = client.client.new_order(symbol=symbol, side='SELL', type='LIMIT', quantity=spending_limit / close_price, price=take_profit)

    logging.info(f'Placed {side} order for {symbol}')
