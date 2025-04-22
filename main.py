import asyncio
import logging
import os
import platform
import signal
import sys
import threading
import time
import traceback

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
    """Handle graceful shutdown on signals like CTRL+C"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")

    # Track shutdown status
    components_shutdown = {
        "scheduler": False,
        "db": False,
        "bot_connection": False,
    }

    # Flag to track if we need to force exit
    force_exit_needed = False

    try:
        # 1. Stop the scheduler first since it depends on the bot for sending alerts
        scheduler = get_scheduler()
        if scheduler and scheduler.running:
            logger.info("Stopping alert scheduler...")
            try:
                scheduler.stop()
                components_shutdown["scheduler"] = True
                logger.info("Alert scheduler stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.info("Scheduler not running or not initialized")
            components_shutdown["scheduler"] = True

        # 2. Close database connections
        logger.info("Closing database connections...")
        try:
            db = get_db()
            db.close()
            components_shutdown["db"] = True
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
            logger.debug(traceback.format_exc())

        # 3. Properly close the bot connection
        if bot and bot.is_ready():
            logger.info("Closing Discord connection...")
            try:
                # Don't try to wait for the close coroutine
                # Discord.py will handle this internally when the process exits
                asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
                logger.info("Discord bot shutdown initiated")
                components_shutdown["bot_connection"] = True
            except Exception as e:
                logger.error(f"Error while closing Discord connection: {e}")
                logger.debug(traceback.format_exc())
                force_exit_needed = True
        else:
            logger.info("Bot not ready, skipping connection cleanup")
            components_shutdown[
                "bot_connection"
            ] = True  # Mark as done since nothing to do

    except Exception as e:
        logger.error(f"Unexpected error during shutdown process: {e}")
        logger.debug(traceback.format_exc())
        force_exit_needed = True

    # Log shutdown status
    logger.info(f"Shutdown status: {components_shutdown}")

    # Final shutdown message
    if all(components_shutdown.values()):
        logger.info("All components shut down successfully")
    else:
        failed_components = [k for k, v in components_shutdown.items() if not v]
        logger.warning(
            f"Some components failed to shut down properly: {failed_components}"
        )
        force_exit_needed = True

    # Exit the process
    if force_exit_needed:
        logger.warning("Forcing exit due to shutdown issues")
        # Use a short delay and then exit to allow logs to flush
        time.sleep(0.5)
        os._exit(1)
    else:
        logger.info("Clean shutdown initiated")
        # Use a short delay and then exit to allow logs to flush
        time.sleep(0.5)
        os._exit(0)


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
            logger.debug(traceback.format_exc())
            sys.exit(1)
