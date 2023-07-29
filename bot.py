#bot.py#
import time
from client import binance_client
import utils
import trading
import logging

logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

def trade_bot(sleep_time=60):
    # Call the function to print the balances
    utils.print_wallet_balances()
    top_coins = trading.get_top_coins(10)  # Get top 10 coins
    logging.info(f'Top coins: {top_coins}')

    while True:
        try:
            # Rebalance portfolio every hour
            if time.time() % 3600 < 1:
                trading.rebalance_portfolio()
                top_coins = trading.get_top_coins(10)  # Get top 10 coins
                logging.info(f'Top coins: {top_coins}')
                balance = binance_client.account()
                total_balance = sum([float(coin['free']) * float(binance_client.ticker_price(coin['asset'] + 'USDT')['price']) for coin in balance['balances'] if coin['asset'] != 'USDT'])
                logging.info(f'Hourly update: Total balance is {total_balance}')
                # Call the function to print the balances
                utils.print_wallet_balances()

            # Get ticker prices for all coins
            symbols = [symbol_info['symbol'] for symbol_info in binance_client.exchange_info()['symbols'] if symbol_info['quoteAsset'] == 'USDT']
            ticker_prices = {ticker['symbol']: float(ticker['lastPrice']) for ticker in [binance_client.ticker_24hr(symbol) for symbol in symbols] if ticker['symbol'].endswith('USDT')}

            # Get account balances
            balances = {balance['asset']: float(balance['free']) for balance in binance_client.account()['balances']}

            # Process each coin
            for coin in top_coins:
                try:
                    trading.process_coin(coin['symbol'], ticker_prices, balances)
                except Exception as e:
                    logging.error(f'Error processing coin {coin}: {str(e)}, Ticker Prices: {ticker_prices}, Balances: {balances}')

            # Sleep for a while to prevent the loop from running too frequently
            time.sleep(sleep_time)

        except Exception as e:
            logging.error(f'Error: {e}')
            break  # Stop the bot if there's an error

trade_bot()
