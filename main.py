import os
import logging
from bot.discord_bot import run_bot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger('trading_alerts')

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

if __name__ == "__main__":
    logger.info("Starting Discord Trading Alerts Bot")
    run_bot()
