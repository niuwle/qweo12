import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()])
