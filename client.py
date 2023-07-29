#client.py#
from binance.spot import Spot
import config

binance_client = Spot(api_key=config.api_key, api_secret=config.api_secret, base_url='https://testnet.binance.vision')

