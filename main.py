import logging
import os
import platform
import signal
import sys
import threading
import time

from bot.db import get_db
from bot.discord_bot import bot, run_bot
from bot.scheduler import get_scheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)
logger = logging.getLogger("trading_alerts")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)


# Setup graceful shutdown
def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")

    # Stop the scheduler if it's running
    scheduler = get_scheduler()
    if scheduler and scheduler.running:
        logger.info("Stopping alert scheduler...")
        scheduler.stop()

    # Close database connections
    logger.info("Closing database connections...")
    db = get_db()
    db.close()

    # Close Discord bot if running
    if bot and bot.is_ready():
        logger.info("Logging out from Discord...")
        # Note: We'd ideally do bot.close() here, but this would need
        # to be handled in the event loop - discord.py will handle this anyway

    logger.info("Shutdown complete")
    sys.exit(0)


def test_shutdown():
    """Test function to verify graceful shutdown works"""
    # Initialize components
    logger.info("Initializing test components...")

    # Get database connection
    db = get_db()

    # Initialize scheduler with a dummy callback
    def dummy_callback(user_id, alerts):
        logger.info(f"Dummy alert callback: {user_id}, {alerts}")

    scheduler = get_scheduler(dummy_callback)
    scheduler.initialize()

    # Start a thread to simulate the bot running
    def run_test():
        logger.info("Test running, press Ctrl+C to test graceful shutdown...")
        try:
            # Run for 60 seconds or until interrupted
            for i in range(60):
                time.sleep(1)
                if i % 10 == 0:
                    logger.info(f"Test running for {i} seconds...")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in test thread")

    thread = threading.Thread(target=run_test)
    thread.daemon = True
    thread.start()

    try:
        # Wait for the thread to complete or for a keyboard interrupt
        thread.join()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    # Register signal handlers
    # Windows doesn't support SIGTERM
    is_windows = platform.system().lower() == "windows"

    # Always register SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # SIGTERM is not supported on Windows
    if not is_windows:
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Registered signal handlers for SIGINT and SIGTERM")
    else:
        logger.info(
            "Registered signal handler for SIGINT (SIGTERM not supported on Windows)"
        )

    # Check if we're running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-shutdown":
        logger.info("Running in test mode to verify graceful shutdown")
        test_shutdown()
    else:
        # Normal bot operation
        logger.info("Starting Discord Trading Alerts Bot")
        try:
            run_bot()
        except KeyboardInterrupt:
            # This should be caught by the signal handler, but just in case
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            sys.exit(1)
