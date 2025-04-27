import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import discord
from discord import app_commands
from dotenv import load_dotenv

from bot.binance import fetch_market_data

# Load our modules
from bot.db import get_db
from bot.scheduler import get_scheduler

try:
    from bot.services.feature_flags import get_flag, reload_flags, set_flag

    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False
    logger.warning("Feature flags module not available")

# Set up logging
# logging.basicConfig is now handled in main.py
logger = logging.getLogger("trading_alerts.discord_bot")
logger.setLevel(logging.DEBUG)  # Ensure we're capturing all logs

# Also ensure that our parent logger captures all logs
root_logger = logging.getLogger("trading_alerts")
root_logger.setLevel(logging.DEBUG)
discord_root_logger = logging.getLogger("discord_trading_alerts")
discord_root_logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in .env file")
    raise ValueError("DISCORD_BOT_TOKEN environment variable is required")

# Valid intervals
VALID_INTERVALS = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
]

# Dictionary of alert explanations
ALERT_EXPLANATIONS = {
    "RSI OVERSOLD": "The Relative Strength Index shows the asset may be oversold (below 30). This could signal a potential buying opportunity as the price might rebound soon.",
    "RSI OVERBOUGHT": "The Relative Strength Index shows the asset may be overbought (above 70). This could signal a potential selling opportunity as the price might drop soon.",
    "MACD BULLISH CROSS": "The Moving Average Convergence Divergence line has crossed above the signal line. This suggests upward momentum may be building, potentially indicating a good time to buy.",
    "MACD BEARISH CROSS": "The Moving Average Convergence Divergence line has crossed below the signal line. This suggests downward momentum may be building, potentially indicating a good time to sell.",
    "EMA BULLISH CROSS": "A shorter-term Exponential Moving Average has crossed above a longer-term EMA. This often signals the beginning of an uptrend and could be a buying opportunity.",
    "EMA BEARISH CROSS": "A shorter-term Exponential Moving Average has crossed below a longer-term EMA. This often signals the beginning of a downtrend and could be a selling opportunity.",
    "UPPER BB BREAKOUT": "Price has broken above the upper Bollinger Band. This suggests strong upward momentum, but the asset might be becoming overbought.",
    "LOWER BB BREAKOUT": "Price has broken below the lower Bollinger Band. This suggests strong downward momentum, but the asset might be becoming oversold.",
    "BOLLINGER SQUEEZE": "The Bollinger Bands have narrowed significantly. This usually precedes a period of high volatility - a big price move may be coming soon, but the direction is uncertain.",
    "VOLUME SPIKE": "Trading volume has suddenly increased well above average. This suggests strong interest in the asset and often precedes significant price movements.",
    "VOLUME Z-SCORE SPIKE": "Trading volume has statistically significant deviation from the norm. This indicates unusual market activity that may lead to price changes.",
    "STRONG BULLISH TREND": "The Average Directional Index (ADX) indicates a strong upward trend is developing. Consider buying as prices may continue rising.",
    "STRONG BEARISH TREND": "The Average Directional Index (ADX) indicates a strong downward trend is developing. Consider selling as prices may continue falling.",
    "TREND REVERSAL to BULLISH": "The trend appears to be changing from downward to upward. This may be an opportunity to buy at the beginning of a new uptrend.",
    "TREND REVERSAL to BEARISH": "The trend appears to be changing from upward to downward. This may be an opportunity to sell before a potential downtrend.",
    "HAMMER PATTERN": "A candlestick pattern showing potential trend reversal from downward to upward. The market may have rejected lower prices, suggesting buying support.",
    "EVENING STAR": "A bearish reversal pattern that appears at the top of an uptrend. It suggests the upward momentum is weakening and prices may start falling.",
    "MORNING STAR": "A bullish reversal pattern that appears at the bottom of a downtrend. It suggests the downward momentum is weakening and prices may start rising.",
    "BULLISH ENGULFING": "A candlestick pattern where a green candle completely engulfs the previous red candle. This suggests buying pressure has overwhelmed selling pressure.",
    "BEARISH ENGULFING": "A candlestick pattern where a red candle completely engulfs the previous green candle. This suggests selling pressure has overwhelmed buying pressure.",
}

# Set up intents
intents = discord.Intents.default()
intents.message_content = True  # Allow the bot to see message content
intents.reactions = True  # Enable reaction events


# Define the bot client
class TradingAlertsBot(discord.Client):
    def __init__(self):
        super().__init__(
            intents=intents,
            chunk_guilds_at_startup=False,  # Optimize startup
            max_messages=None,  # Store all messages the bot sees
        )
        self.tree = app_commands.CommandTree(self)
        self.db = get_db()
        self.alert_channels: Dict[
            str, List[discord.TextChannel]
        ] = {}  # user_id -> list of channels
        self.synced = False
        self.scheduler = None
        self.alert_messages = (
            {}
        )  # Store message_id -> alert_type mapping for reaction handling

    async def setup_hook(self):
        """Set up the bot's background tasks and sync commands"""
        # Start the scheduler with our notification callback
        self.scheduler = get_scheduler(self.send_alert_notification)
        self.scheduler.initialize()

        # Explicitly log before syncing to show all command registrations
        logger.info("Checking registered commands before sync...")
        commands = self.tree.get_commands()
        command_names = [cmd.name for cmd in commands]
        logger.info(f"Commands registered in code: {', '.join(command_names)}")

        # Make sure purge commands are removed
        for cmd in list(self.tree.get_commands()):
            if cmd.name in ["purge", "purge_force"]:
                # Remove any purge commands from the command tree
                self.tree._remove_command(cmd)
                logger.info(f"Removed command '{cmd.name}' from command tree")

        # Sync commands with Discord - ensure this happens on startup
        try:
            logger.info("Syncing commands with Discord globally...")
            await self.tree.sync()
            commands = await self.tree.fetch_commands()
            logger.info(f"Successfully synced {len(commands)} global commands")
            command_names = [cmd.name for cmd in commands]
            logger.info(f"Available commands: {', '.join(command_names)}")
            self.synced = True
        except Exception as e:
            logger.error(f"Error syncing commands: {e}")
            self.synced = False

        # Start background task to periodically clean up old alert messages
        self.bg_tasks = []
        cleanup_task = self.loop.create_task(self.periodic_cleanup_task())
        self.bg_tasks.append(cleanup_task)
        logger.info("Started background cleanup task")

    async def on_ready(self):
        """Called when the bot has connected to Discord"""
        logger.info(f"{self.user} has connected to Discord!")

        # Sync commands to all joined guilds for immediate availability
        logger.info(f"{self.user} ready — syncing commands to all joined guilds…")
        for guild in self.guilds:
            try:
                await self.tree.sync(guild=guild)
                logger.info(f" • synced to {guild.name} ({guild.id})")
            except Exception as e:
                logger.error(f" ! failed to sync to {guild.id}: {e}")

        # Load alert channels from database
        await self.load_alert_channels()

        # Set up status message - discord.py doesn't fully support CustomActivity, use this instead
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.custom,
                name="📊 Tracking crypto signals",
                state="📊 Tracking crypto signals",
            )
        )

        # Send test alerts (temporary)
        # asyncio.create_task(self.send_test_alerts())

    async def on_guild_join(self, guild):
        """Called when the bot joins a new guild (server)"""
        logger.info(f"Bot joined guild: {guild.name} (ID: {guild.id})")

        # Try to find a general or bot channel to send welcome message
        channel = None
        for ch in guild.text_channels:
            if ch.permissions_for(guild.me).send_messages:
                if ch.name.lower() in [
                    "general",
                    "bot",
                    "bot-commands",
                    "commands",
                    "alerts",
                ]:
                    channel = ch
                    break

        # If we found a channel, send welcome message
        if channel:
            await self.send_welcome_message(channel)

    async def send_welcome_message(self, channel):
        """Send a welcome message with instructions"""
        embed = discord.Embed(
            title="🚀 Crypto Trading Alerts Bot",
            description="Thanks for adding me to your server! I'll send alerts when technical indicators signal trading opportunities.",
            color=discord.Color.blue(),
        )

        embed.add_field(
            name="Getting Started",
            value="Use `/watch BTCUSDT` to start monitoring Bitcoin\n"
            "Use `/list` to see your watched pairs\n"
            "Use `/settings` to customize your alerts",
            inline=False,
        )

        embed.add_field(
            name="Available Commands",
            value="`/watch symbol [interval]` - Add a trading pair to watchlist\n"
            "`/unwatch symbol [interval]` - Remove a pair from watchlist\n"
            "`/list` - Show your watchlist\n"
            "`/settings` - View your alert settings\n"
            "`/alerts` - Enable or disable specific alert types\n"
            "`/stats` - View alert statistics\n"
            "`/guide` - Get guidance on optimal timeframes\n"
            "`/help` - Show this help information\n"
            "`/cleanup` - Delete old alert messages",
            inline=False,
        )

        embed.add_field(
            name="Tip",
            value="Click the ❓ reaction on alerts to get an explanation of what the alert means.",
            inline=False,
        )

        embed.add_field(
            name="Credits",
            value="Made by [gabrielsaban](https://github.com/gabrielsaban)\n"
            "[View Project on GitHub](https://github.com/gabrielsaban/discord-trading-alerts)",
            inline=False,
        )

        embed.set_footer(text="React with 📌 to pin this message for future reference")

        message = await channel.send(embed=embed)
        await message.add_reaction("📌")  # Pin reaction

    async def send_alert_notification(self, user_id: str, alerts: List[str]):
        """Send alert notifications to the specified user"""
        # Skip if no channels are registered for this user
        if user_id not in self.alert_channels or not self.alert_channels[user_id]:
            logger.warning(f"No channels registered for user {user_id}")
            return

        # Get user data
        user = self.db.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found in database")
            return

        # Safety check to make sure we're running in an asyncio task
        try:
            asyncio.current_task()
        except RuntimeError:
            logger.warning(
                "send_alert_notification called outside of an asyncio task context"
            )
            return

        # Send alerts to all registered channels
        for channel in self.alert_channels[user_id]:
            try:
                # Try to get a fresh channel reference to avoid stale references
                current_channel = channel
                if hasattr(self, "get_channel"):
                    fetched_channel = self.get_channel(channel.id)
                    if fetched_channel is not None:
                        current_channel = fetched_channel

                # Process each alert
                for alert in alerts:
                    try:
                        # Extract symbol from alert message
                        symbol = (
                            alert.split(":")[1].split()[0]
                            if ":" in alert
                            else "Unknown"
                        )

                        # Get interval from the alert message using a parameter
                        # Format should be: alert_message | interval
                        alert_parts = alert.split("|")
                        interval = "Unknown"
                        if len(alert_parts) >= 2:
                            # Extract interval from the second part
                            interval = alert_parts[1].strip()
                        else:
                            # Fallback to database lookup
                            watchlist = self.db.get_user_watchlist(user_id)
                            for item in watchlist:
                                if item["symbol"] == symbol:
                                    interval = item["interval"]
                                    break

                        # Use only the alert message part without the interval
                        clean_alert = alert_parts[0].strip()

                        # Add user mention and handle the pipe separator
                        user_mention = f"<@{user_id}>"
                        
                        # Detect if this is a batch summary
                        is_batch_summary = "ALERT SUMMARY" in clean_alert
                        
                        if is_batch_summary:
                            # For batch summaries, we need special handling
                            # Format the reformatted alert with user mention
                            reformatted_alert = f"{user_mention}\n{clean_alert}"
                            
                            # Count how many alerts are in the summary
                            alert_count = 0
                            # Look for lines like "1.", "2.", etc.
                            for line in clean_alert.split("\n"):
                                if line.strip().startswith(("1.", "2.", "3.")):
                                    alert_count += 1

                            # Extract the total count from the "more signals" line
                            more_signals = 0
                            for line in clean_alert.split("\n"):
                                if "more signals not shown" in line:
                                    try:
                                        more_signals = int(
                                            line.split("_")[1].split()[0]
                                        )
                                    except:
                                        pass

                            total_alerts = alert_count + more_signals
                            title = f"🔔 Batch Summary: {symbol} ({total_alerts} alerts)"
                        elif "\nPrice: " in clean_alert:
                            # Regular alert with price
                            parts = clean_alert.split("\nPrice: ")
                            alert_header = parts[0]
                            price_and_details = parts[1]

                            # Check if there are detailed threshold values after the price
                            if "\n" in price_and_details:
                                price_part, details_part = price_and_details.split(
                                    "\n", 1
                                )
                                reformatted_alert = f"{user_mention}\n{alert_header}\nPrice: {price_part}\n{details_part}"
                                logger.debug(
                                    f"Alert with detailed threshold info: {details_part}"
                                )
                            else:
                                reformatted_alert = f"{user_mention}\n{alert_header}\nPrice: {price_and_details}"
                        else:
                            # Regular alert without price
                            reformatted_alert = f"{user_mention}\n{clean_alert}"

                        # Set title for regular alerts
                        if not is_batch_summary:
                            title = f"⚠️ Alert: {symbol} ({interval})"

                        # Create embed with custom title
                        embed = discord.Embed(
                            title=title,
                            description=reformatted_alert,
                            color=self._get_color_for_alert(alert),
                        )

                        # Add timestamp
                        embed.timestamp = discord.utils.utcnow()

                        # Simple direct send without any nested coroutines or tasks
                        try:
                            message = await current_channel.send(embed=embed)
                            logger.info(
                                f"Sent alert for {symbol} ({interval}) to channel {current_channel.id}"
                            )

                            # Add only question mark reaction for explanation
                            await message.add_reaction("❓")

                            # Extract alert type for explanations
                            alert_type = ""
                            if "**" in alert:
                                parts = alert.split("**")
                                if len(parts) >= 3:
                                    alert_type = parts[
                                        1
                                    ]  # Get text between first set of **

                            # Store message ID and alert type
                            self.alert_messages[message.id] = alert_type

                        except Exception as e:
                            logger.error(
                                f"Failed to send alert to channel {current_channel.id}: {e}"
                            )

                        # Add a small delay between alerts to avoid rate limiting
                        await asyncio.sleep(1.0)

                    except Exception as e:
                        logger.error(
                            f"Error processing alert for channel {current_channel.id}: {e}"
                        )
            except Exception as e:
                logger.error(f"Error accessing channel for user {user_id}: {e}")
                import traceback

                logger.error(f"Alert error details: {traceback.format_exc()}")

    def _get_color_for_alert(self, alert: str) -> discord.Color:
        """Get the appropriate color for an alert based on its type"""
        if "🟢" in alert or "BULLISH" in alert:
            return discord.Color.green()
        elif "🔴" in alert or "BEARISH" in alert:
            return discord.Color.red()
        elif "🟡" in alert or "SQUEEZE" in alert:
            return discord.Color.gold()
        else:
            return discord.Color.blue()

    async def load_alert_channels(self):
        """Load alert channels from the database"""
        try:
            # Clear existing channels
            self.alert_channels = {}

            # Get all active symbols to find their users
            active_symbols = self.db.get_all_active_symbols()
            unique_users = set()

            for symbol, interval in active_symbols:
                users = self.db.get_users_watching_symbol(symbol, interval)
                unique_users.update(users)

            # Load channels for each user
            for user_id in unique_users:
                channels_data = self.db.get_user_alert_channels(user_id)

                if not channels_data:
                    logger.warning(f"No alert channels found for user {user_id}")
                    continue

                self.alert_channels[user_id] = []

                for channel_data in channels_data:
                    channel_id = int(channel_data["channel_id"])
                    guild_id = int(channel_data["guild_id"])

                    # Try to get the guild
                    guild = self.get_guild(guild_id)
                    if not guild:
                        logger.warning(f"Guild {guild_id} not found for user {user_id}")
                        continue

                    # Try to get the channel
                    channel = guild.get_channel(channel_id)
                    if not channel:
                        logger.warning(
                            f"Channel {channel_id} not found in guild {guild_id}"
                        )
                        continue

                    # Add channel to user's alert channels
                    self.alert_channels[user_id].append(channel)
                    logger.info(f"Loaded alert channel {channel_id} for user {user_id}")

            logger.info(f"Loaded alert channels for {len(self.alert_channels)} users")
        except Exception as e:
            logger.error(f"Error loading alert channels: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def register_channel(self, user_id: str, channel: discord.TextChannel):
        """Register a channel for a user's alerts"""
        if user_id not in self.alert_channels:
            self.alert_channels[user_id] = []

        if channel not in self.alert_channels[user_id]:
            self.alert_channels[user_id].append(channel)
            # Also save to database
            self.db.register_alert_channel(
                user_id, str(channel.id), str(channel.guild.id)
            )
            logger.info(f"Registered channel {channel.id} for user {user_id}")

    def unregister_channel(self, user_id: str, channel: discord.TextChannel):
        """Unregister a channel for a user's alerts"""
        if user_id in self.alert_channels and channel in self.alert_channels[user_id]:
            self.alert_channels[user_id].remove(channel)
            # Also remove from database
            self.db.unregister_alert_channel(user_id, str(channel.id))
            logger.info(f"Unregistered channel {channel.id} for user {user_id}")

    async def send_test_alerts(self):
        """Send examples of all possible alert types to a specific user and channel"""
        # Define the target user ID and channel ID
        user_id = "153242761531752449"
        channel_id = 1363459585435897929

        # Make sure the bot is ready before proceeding
        if not self.is_ready():
            logger.warning("Bot not ready yet, waiting before sending test alerts...")
            await asyncio.sleep(5)

        # Try to get the channel
        channel = self.get_channel(channel_id)
        if not channel:
            logger.error(f"Could not find channel with ID {channel_id}")
            return

        # Register the channel for the user
        self.db.create_user(user_id, "Test User")
        self.register_channel(user_id, channel)

        # Define example alerts for all types
        symbol = "BTCUSDT"
        interval = "15m"  # Use 15m as test interval
        price = "42,069.42"
        user_mention = f"<@{user_id}>"
        test_alerts = [
            # RSI alerts
            f"{user_mention}\n🔴 **RSI OVERSOLD**: {symbol} RSI at 29.8\nPrice: {price}",
            f"{user_mention}\n🟢 **RSI OVERBOUGHT**: {symbol} RSI at 70.5\nPrice: {price}",
            # MACD alerts
            f"{user_mention}\n🟢 **MACD BULLISH CROSS**: {symbol}\nPrice: {price}",
            f"{user_mention}\n🔴 **MACD BEARISH CROSS**: {symbol}\nPrice: {price}",
            # EMA alerts
            f"{user_mention}\n🟢 **EMA BULLISH CROSS**: {symbol} EMA9 crossed above EMA21\nPrice: {price}",
            f"{user_mention}\n🔴 **EMA BEARISH CROSS**: {symbol} EMA9 crossed below EMA21\nPrice: {price}",
            # Bollinger Band alerts
            f"{user_mention}\n🟢 **UPPER BB BREAKOUT**: {symbol} Price broke above upper band\nPrice: {price}",
            f"{user_mention}\n🔴 **LOWER BB BREAKOUT**: {symbol} Price broke below lower band\nPrice: {price}",
            f"{user_mention}\n🟡 **BOLLINGER SQUEEZE**: {symbol} Bands narrowing, potential breakout\nPrice: {price}",
            # Volume alerts
            f"{user_mention}\n📊 **VOLUME SPIKE**: {symbol} Volume 3.2x above average\nPrice: {price}",
            # ADX alerts
            f"{user_mention}\n📏 **STRONG BULLISH 📈 TREND**: {symbol} ADX: 28.5\nPrice: {price}",
            f"{user_mention}\n📏 **STRONG BEARISH 📉 TREND**: {symbol} ADX: 26.8\nPrice: {price}",
            f"{user_mention}\n🔄 **TREND REVERSAL to BULLISH 📈**: {symbol} ADX: 30.2\nPrice: {price}",
            f"{user_mention}\n🔄 **TREND REVERSAL to BEARISH 📉**: {symbol} ADX: 29.7\nPrice: {price}",
            # Pattern alerts
            f"{user_mention}\n🔨 **HAMMER PATTERN**: {symbol} Potential reversal\nPrice: {price}",
            f"{user_mention}\n⭐ **EVENING STAR**: {symbol} Potential bearish reversal\nPrice: {price}",
            f"{user_mention}\n⭐ **MORNING STAR**: {symbol} Potential bullish reversal\nPrice: {price}",
            f"{user_mention}\n🔄 **BULLISH ENGULFING**: {symbol}\nPrice: {price}",
            f"{user_mention}\n🔄 **BEARISH ENGULFING**: {symbol}\nPrice: {price}",
        ]

        logger.info(f"Sending {len(test_alerts)} test alerts to channel {channel_id}")

        # Send each alert with a delay to avoid rate limiting
        for alert in test_alerts:
            try:
                # Extract symbol from alert message
                alert_symbol = (
                    alert.split(":")[1].split()[0] if ":" in alert else "Unknown"
                )

                # Create embed with interval
                embed = discord.Embed(
                    title=f"⚠️ Alert: {alert_symbol} *({interval})*",
                    description=alert,
                    color=self._get_color_for_alert(alert),
                )

                # Add timestamp
                embed.timestamp = discord.utils.utcnow()

                # Send the message
                message = await channel.send(embed=embed)
                logger.info(f"Sent test alert: {alert[:30]}...")

                # Add reactions: checkmark to delete and question mark for explanation
                await message.add_reaction("✅")
                await message.add_reaction("❓")

                # Extract alert type for explanations
                alert_type = ""
                if "**" in alert:
                    parts = alert.split("**")
                    if len(parts) >= 3:
                        alert_type = parts[1]  # Get text between first set of **

                # Store message ID and alert type
                self.alert_messages[message.id] = alert_type

                # Add delay between messages
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error sending test alert: {e}")

        logger.info("Finished sending all test alerts")

    async def periodic_cleanup_task(self):
        """Background task to periodically clean up old alert messages"""
        # Wait until the bot is ready before starting cleanup
        await self.wait_until_ready()
        logger.info("Periodic cleanup task ready")

        # Clean up every 4 hours - adjust this interval as needed
        CLEANUP_INTERVAL = 4 * 60 * 60  # 4 hours in seconds

        while not self.is_closed():
            try:
                # Get current time for age calculations
                now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator

                # Track total messages cleaned up
                total_cleaned = 0

                # Check each channel for alert messages older than 12 hours
                if self.alert_channels:
                    for user_id, channels in self.alert_channels.items():
                        for channel in channels:
                            try:
                                # Try to get a fresh channel reference
                                current_channel = channel
                                if hasattr(self, "get_channel"):
                                    fetched_channel = self.get_channel(channel.id)
                                    if fetched_channel is not None:
                                        current_channel = fetched_channel

                                # Only check a reasonable number of messages to avoid rate limits
                                messages_to_delete = []

                                async for message in current_channel.history(limit=30):
                                    # Skip messages less than 12 hours old
                                    message_age = now - message.created_at.replace(
                                        tzinfo=None
                                    )
                                    if (
                                        message_age.total_seconds() < 12 * 60 * 60
                                    ):  # 12 hours
                                        continue

                                    # Check if this is one of our alert messages
                                    if (
                                        message.author.id == self.user.id
                                        and message.embeds
                                        and len(message.embeds) > 0
                                        and message.embeds[0].title
                                        and "Alert:" in message.embeds[0].title
                                    ):
                                        messages_to_delete.append(message)

                                # Delete old messages
                                for message in messages_to_delete:
                                    try:
                                        await message.delete()
                                        total_cleaned += 1

                                        # Remove from tracked messages if present
                                        if message.id in self.alert_messages:
                                            del self.alert_messages[message.id]

                                        # Add a longer delay to avoid rate limits (Discord's rate limit is 1 per second)
                                        await asyncio.sleep(1.1)
                                    except Exception as e:
                                        logger.debug(f"Error deleting old message: {e}")

                                if messages_to_delete:
                                    logger.info(
                                        f"Auto-cleaned {len(messages_to_delete)} old alert messages from {current_channel.id}"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Error in periodic cleanup for channel: {e}"
                                )

                if total_cleaned > 0:
                    logger.info(
                        f"Periodic cleanup complete: deleted {total_cleaned} old alert messages"
                    )

            except Exception as e:
                logger.error(f"Error in periodic cleanup task: {e}")

            # Wait for next cleanup interval
            await asyncio.sleep(CLEANUP_INTERVAL)

    async def on_raw_reaction_remove(self, payload):
        """Handle when users remove reactions, even if the message isn't in cache"""
        # Skip if this is our own reaction being removed
        if payload.user_id == self.user.id:
            return

        logger.info(
            f"Raw reaction remove: {payload.emoji.name} by user {payload.user_id} on message {payload.message_id}"
        )

        # We only care about question mark reactions on alert messages
        if payload.emoji.name != "❓":
            return

        # Get the channel and message
        try:
            channel = self.get_channel(payload.channel_id)
            if not channel:
                logger.warning(f"Channel {payload.channel_id} not found")
                return

            # Get the message if we need to
            message = await channel.fetch_message(payload.message_id)
            if not message or message.author.id != self.user.id:
                logger.warning(
                    f"Message {payload.message_id} not found or not from bot"
                )
                return

            # Check if this is an alert message by examining its embed
            is_alert = (
                message.embeds
                and len(message.embeds) > 0
                and message.embeds[0].title
                and "Alert:" in message.embeds[0].title
            )

            if not is_alert:
                return

            # Check if there are any question mark reactions left from users (excluding the bot)
            has_user_question_mark = False

            # Check all reactions on the message
            for reaction in message.reactions:
                if reaction.emoji == "❓":
                    # Get all users who reacted
                    users = [u async for u in reaction.users()]
                    non_bot_users = [u for u in users if u.id != self.user.id]
                    logger.info(
                        f"Question mark reaction has {len(users)} users ({len(non_bot_users)} non-bot)"
                    )

                    # If we have any non-bot users, keep the explanation
                    if non_bot_users:
                        has_user_question_mark = True
                        break

            # If no more user question mark reactions, remove the explanation
            if not has_user_question_mark:
                logger.info(
                    f"No more user question mark reactions, removing explanation"
                )

                # Get the original embed
                embed = message.embeds[0] if message.embeds else None

                if not embed:
                    logger.warning("No embed found in message")
                    return

                # Find and remove the Explanation field
                new_fields = []
                has_explanation = False

                for field in embed.fields:
                    if field.name != "Explanation":
                        new_fields.append(field)
                    else:
                        has_explanation = True

                logger.info(
                    f"Found explanation field: {has_explanation}, fields before: {len(embed.fields)}, after: {len(new_fields)}"
                )

                # If no fields were removed, no need to update
                if len(new_fields) == len(embed.fields):
                    logger.info("No explanation field to remove")
                    return

                # Create a new embed with the same properties but without the explanation field
                new_embed = discord.Embed(
                    title=embed.title,
                    description=embed.description,
                    color=embed.color,
                    timestamp=embed.timestamp,
                )

                # Add remaining fields
                for field in new_fields:
                    new_embed.add_field(
                        name=field.name, value=field.value, inline=field.inline
                    )

                # Copy the footer if it exists
                if embed.footer:
                    new_embed.set_footer(
                        text=embed.footer.text, icon_url=embed.footer.icon_url
                    )

                # Update the message
                try:
                    await message.edit(embed=new_embed)
                    logger.info(f"Removed explanation from alert message {message.id}")
                except Exception as e:
                    logger.error(f"Error updating message to remove explanation: {e}")

        except Exception as e:
            logger.error(f"Error handling raw reaction remove: {e}")

    async def on_raw_reaction_add(self, payload):
        """Handle when users add reactions, even if the message isn't in cache"""
        # Skip if this is our own reaction being added
        if payload.user_id == self.user.id:
            return

        logger.info(
            f"Raw reaction add: {payload.emoji.name} by user {payload.user_id} on message {payload.message_id}"
        )

        # Get the channel and message
        try:
            channel = self.get_channel(payload.channel_id)
            if not channel:
                logger.warning(f"Channel {payload.channel_id} not found")
                return

            # Get the message
            message = await channel.fetch_message(payload.message_id)
            if not message or message.author.id != self.user.id:
                logger.warning(
                    f"Message {payload.message_id} not found or not from bot"
                )
                return

            # Handle pin reaction on welcome message
            if (
                payload.emoji.name == "📌"
                and message.embeds
                and "Crypto Trading Alerts Bot" in message.embeds[0].title
            ):
                # Get user and channel permissions
                user = await self.fetch_user(payload.user_id)
                channel_permissions = channel.permissions_for(user)
                bot_permissions = channel.permissions_for(message.guild.me)

                # Check if the bot has permissions to pin
                if not bot_permissions.manage_messages:
                    try:
                        # Inform about missing permissions
                        await channel.send(
                            f"I don't have permission to pin messages. Please give me the 'Manage Messages' permission.",
                            delete_after=10,
                        )
                    except Exception:
                        pass
                    return

                # Only server mods/admins or users with pin permissions can pin the message
                if (
                    channel_permissions.manage_messages
                    or channel_permissions.administrator
                ):
                    try:
                        # Pin the message
                        await message.pin()
                        logger.info(f"Pinned welcome message in channel {channel.id}")

                        # Remove the pin reaction to indicate it's been pinned
                        await message.remove_reaction("📌", user)
                    except discord.Forbidden:
                        logger.error("Bot doesn't have permission to pin messages")
                    except Exception as e:
                        logger.error(f"Error pinning message: {e}")
                else:
                    # Let user know they don't have permission to pin
                    try:
                        await channel.send(
                            f"<@{payload.user_id}> You don't have permission to pin messages in this channel.",
                            delete_after=5,
                        )
                        # Remove their reaction
                        await message.remove_reaction("📌", user)
                    except Exception:
                        pass

                return  # Welcome messages aren't alert messages, so return here

            # Check if this is an alert message by examining its embed
            is_alert = (
                message.embeds
                and len(message.embeds) > 0
                and message.embeds[0].title
                and "Alert:" in message.embeds[0].title
            )

            if not is_alert:
                return

            # Handle question mark reaction (add explanation)
            if payload.emoji.name == "❓":
                # Check if we already added the explanation (to avoid adding it multiple times)
                if any(
                    field.name == "Explanation" for field in message.embeds[0].fields
                ):
                    logger.info(f"Explanation already exists, not adding again")
                    return

                # Extract alert type from message content to find the right explanation
                alert_type = None
                embed_description = message.embeds[0].description

                if embed_description and "**" in embed_description:
                    # Try to extract the alert type from between ** markers
                    parts = embed_description.split("**")
                    if len(parts) >= 3:
                        alert_type = parts[
                            1
                        ].strip()  # Get text between first set of **

                # Find matching explanation
                explanation = None
                if alert_type:
                    for key, value in ALERT_EXPLANATIONS.items():
                        if key in alert_type:
                            explanation = value
                            break

                if not explanation:
                    explanation = "This alert suggests a potential trading opportunity. For more details on technical analysis, please research the specific indicator mentioned."

                # Add explanation field
                embed = message.embeds[0]
                embed.add_field(name="Explanation", value=explanation, inline=False)

                # Update the message
                try:
                    await message.edit(embed=embed)
                    logger.info(f"Added explanation to alert message {message.id}")

                    # Make sure the bot also has a reaction to keep the explanation
                    # Check if the bot already has a reaction
                    bot_has_reaction = False
                    for r in message.reactions:
                        if r.emoji == "❓":
                            async for u in r.users():
                                if u.id == self.user.id:
                                    bot_has_reaction = True
                                    break
                            if bot_has_reaction:
                                break

                    # Add bot's reaction if needed
                    if not bot_has_reaction:
                        await message.add_reaction("❓")
                        logger.info(
                            f"Added bot's question mark reaction to keep explanation visible"
                        )
                except Exception as e:
                    logger.error(f"Error updating message with explanation: {e}")

        except Exception as e:
            logger.error(f"Error handling raw reaction add: {e}")


# Create bot instance
bot = TradingAlertsBot()


# Watch command
@bot.tree.command(name="watch", description="Add a trading pair to your watchlist")
@app_commands.describe(
    symbol="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    interval="Time interval for candles (default: 15m)",
    channel="Channel to send alerts to (default: current channel)",
)
@app_commands.choices(
    interval=[
        app_commands.Choice(name=interval, value=interval)
        for interval in VALID_INTERVALS
    ]
)
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def watch_command(
    interaction: discord.Interaction,
    symbol: str,
    interval: str = "15m",
    channel: Optional[discord.TextChannel] = None,
):
    """Add a trading pair to the user's watchlist"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)
    symbol = symbol.upper()  # Standardize to uppercase

    # Validate interval
    if interval not in VALID_INTERVALS:
        await interaction.followup.send(
            f"Invalid interval: {interval}. Valid intervals are: {', '.join(VALID_INTERVALS)}",
            ephemeral=True,
        )
        return

    # Validate the symbol by attempting to fetch data
    try:
        df = fetch_market_data(symbol=symbol, interval=interval, limit=5)
        if df is None or df.empty:
            await interaction.followup.send(
                f"Could not fetch data for {symbol}. Please check that the symbol is valid on Binance.",
                ephemeral=True,
            )
            return
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {e}")
        await interaction.followup.send(
            f"Error validating {symbol}: {str(e)}. Please check that the symbol is valid on Binance.",
            ephemeral=True,
        )
        return

    # Get or create user
    user = bot.db.get_user(user_id)
    if not user:
        bot.db.create_user(user_id, interaction.user.display_name, interaction.user.id)

    # Check if symbol is already in watchlist
    watchlist = bot.db.get_user_watchlist(user_id)
    already_watching = any(
        w["symbol"] == symbol and w["interval"] == interval for w in watchlist
    )

    # Add to watchlist
    success = bot.db.add_to_watchlist(user_id, symbol, interval)

    if success:
        # Register the channel for alerts
        alert_channel = channel or interaction.channel
        bot.register_channel(user_id, alert_channel)

        # Add to scheduler
        bot.scheduler.add_symbol(symbol, interval)

        # Notify user
        if already_watching:
            await interaction.followup.send(
                f"✅ You're already watching {symbol} ({interval}). Updated alert channel to {alert_channel.mention}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                f"🔔 Added {symbol} ({interval}) to your watchlist. Alerts will be sent to {alert_channel.mention}.",
                ephemeral=True,
            )
    else:
        await interaction.followup.send(
            f"Failed to add {symbol} to your watchlist. Please try again later.",
            ephemeral=True,
        )


# Unwatch command
@bot.tree.command(
    name="unwatch", description="Remove a trading pair from your watchlist"
)
@app_commands.describe(
    symbol="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    interval="Time interval for candles (default: 15m)",
)
@app_commands.choices(
    interval=[
        app_commands.Choice(name=interval, value=interval)
        for interval in VALID_INTERVALS
    ]
)
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def unwatch_command(
    interaction: discord.Interaction, symbol: str, interval: str = "15m"
):
    """Remove a trading pair from the user's watchlist"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)
    symbol = symbol.upper()  # Standardize to uppercase

    # Validate interval
    if interval not in VALID_INTERVALS:
        await interaction.followup.send(
            f"Invalid interval: {interval}. Valid intervals are: {', '.join(VALID_INTERVALS)}",
            ephemeral=True,
        )
        return

    # Remove from watchlist
    success = bot.db.remove_from_watchlist(user_id, symbol, interval)

    if success:
        # Check if any users are still watching this symbol/interval
        users_watching = bot.db.get_users_watching_symbol(symbol, interval)

        # If no users are watching, remove the symbol from the scheduler
        if not users_watching:
            bot.scheduler.remove_symbol(symbol, interval)

        await interaction.followup.send(
            f"🔕 Removed {symbol} ({interval}) from your watchlist.", ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"{symbol} ({interval}) is not in your watchlist.", ephemeral=True
        )


# List command
@bot.tree.command(name="list", description="Show your watched trading pairs")
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def list_command(interaction: discord.Interaction):
    """Show the user's watchlist"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)

    # Get watchlist
    watchlist = bot.db.get_user_watchlist(user_id)

    if not watchlist:
        await interaction.followup.send(
            "Your watchlist is empty. Use `/watch` to add trading pairs.",
            ephemeral=True,
        )
        return

    # Create embed
    embed = discord.Embed(
        title="🔍 Your Watchlist",
        description=f"You are watching {len(watchlist)} trading pairs:",
        color=discord.Color.blue(),
    )

    # Group by interval
    intervals = {}
    for item in watchlist:
        interval = item["interval"]
        if interval not in intervals:
            intervals[interval] = []
        intervals[interval].append(item["symbol"])

    # Add fields for each interval
    for interval, symbols in intervals.items():
        embed.add_field(
            name=f"{interval} Interval",
            value=", ".join(symbols) or "None",
            inline=False,
        )

    await interaction.followup.send(embed=embed, ephemeral=True)


# Settings command
@bot.tree.command(name="settings", description="View your alert settings (custom thresholds coming soon)")
@app_commands.describe(setting="Setting name", value="Setting value")
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def settings_command(
    interaction: discord.Interaction,
    setting: Optional[str] = None,
    value: Optional[str] = None,
):
    """View alert settings"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)

    # Get user settings
    user = bot.db.get_user(user_id)
    if not user:
        bot.db.create_user(user_id, interaction.user.display_name, interaction.user.id)
        user = bot.db.get_user(user_id)

    # Show current settings
    embed = discord.Embed(
        title="⚙️ Your Alert Settings",
        description="Custom alert thresholds will be available in a future premium tier. Currently, all alerts use default thresholds.",
        color=discord.Color.blue(),
    )

    settings = user["settings"]

    embed.add_field(
        name="Current Default Thresholds",
        value=f"• RSI: Oversold (<30), Overbought (>70)\n"
        f"• EMA: Short (9), Long (21)\n"
        f"• Volume Spike: 2.5x average\n"
        f"• Bollinger Squeeze: 0.05\n"
        f"• ADX: 25",
        inline=False,
    )

    # Show enabled alert types
    enabled = settings.get(
        "enabled_alerts", ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"]
    )
    embed.add_field(
        name="Enabled Alerts", value=", ".join(enabled) or "None", inline=False
    )
    
    embed.add_field(
        name="Alert Configuration",
        value="Use the `/alerts` command to enable or disable specific alert types.",
        inline=False,
    )

    await interaction.followup.send(embed=embed, ephemeral=True)


# Enable/disable alerts command
@bot.tree.command(name="alerts", description="Enable or disable specific alert types")
@app_commands.describe(
    action="Enable or disable alerts", alert_type="Type of alert to enable/disable"
)
@app_commands.choices(
    action=[
        app_commands.Choice(name="Enable", value="enable"),
        app_commands.Choice(name="Disable", value="disable"),
    ],
    alert_type=[
        app_commands.Choice(name="RSI (Relative Strength Index)", value="rsi"),
        app_commands.Choice(
            name="MACD (Moving Average Convergence Divergence)", value="macd"
        ),
        app_commands.Choice(name="EMA Crossovers", value="ema"),
        app_commands.Choice(name="Bollinger Bands", value="bb"),
        app_commands.Choice(name="Volume Spikes", value="volume"),
        app_commands.Choice(name="ADX (Average Directional Index)", value="adx"),
        app_commands.Choice(name="Candlestick Patterns", value="pattern"),
        app_commands.Choice(name="All Alerts", value="all"),
    ],
)
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def alerts_command(
    interaction: discord.Interaction, action: str, alert_type: str
):
    """Enable or disable specific alert types"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)

    # Get user settings
    user = bot.db.get_user(user_id)
    if not user:
        bot.db.create_user(user_id, interaction.user.display_name, interaction.user.id)
        user = bot.db.get_user(user_id)

    # Get enabled alerts
    settings = user["settings"]
    enabled_alerts = settings.get(
        "enabled_alerts", ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"]
    )

    # Update enabled alerts
    all_types = ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"]

    if action == "enable":
        if alert_type == "all":
            enabled_alerts = all_types.copy()
        elif alert_type not in enabled_alerts:
            enabled_alerts.append(alert_type)
    else:  # disable
        if alert_type == "all":
            enabled_alerts = []
        elif alert_type in enabled_alerts:
            enabled_alerts.remove(alert_type)

    # Update settings
    bot.db.update_user_settings(user_id, {"enabled_alerts": enabled_alerts})

    # Confirm to user
    if action == "enable":
        message = f"✅ Enabled {alert_type} alerts."
    else:
        message = f"🚫 Disabled {alert_type} alerts."

    if alert_type == "all":
        if action == "enable":
            message = "✅ Enabled all alert types."
        else:
            message = "🚫 Disabled all alert types."

    await interaction.followup.send(
        f"{message}\nCurrently enabled: {', '.join(enabled_alerts) or 'None'}",
        ephemeral=True,
    )


# Stats command
@bot.tree.command(name="stats", description="View statistics about your alerts")
@app_commands.describe(days="Number of days to include in the stats (default: 7)")
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def stats_command(interaction: discord.Interaction, days: int = 7):
    """View alert statistics"""
    await interaction.response.defer(ephemeral=True)

    user_id = str(interaction.user.id)

    # Validate days
    if days < 1 or days > 30:
        await interaction.followup.send(
            f"Invalid number of days: {days}. Must be between 1 and 30.", ephemeral=True
        )
        return

    # Get recent alerts
    alerts = bot.db.get_recent_alerts(user_id, hours=days * 24)

    if not alerts:
        await interaction.followup.send(
            f"No alerts in the past {days} days. Add trading pairs with `/watch` to get started!",
            ephemeral=True,
        )
        return

    # Create stats embed
    embed = discord.Embed(
        title=f"📊 Your Alert Stats (Past {days} days)",
        description=f"You've received {len(alerts)} alerts:",
        color=discord.Color.blue(),
    )

    # Count by type
    alert_types = {}
    symbols = {}

    for alert in alerts:
        # Count by type
        alert_type = alert["alert_type"]
        if alert_type not in alert_types:
            alert_types[alert_type] = 0
        alert_types[alert_type] += 1

        # Count by symbol
        symbol = alert["symbol"]
        if symbol not in symbols:
            symbols[symbol] = 0
        symbols[symbol] += 1

    # Add alert types field
    alert_type_text = ""
    for alert_type, count in sorted(
        alert_types.items(), key=lambda x: x[1], reverse=True
    ):
        alert_type_text += f"{alert_type.upper()}: {count}\n"

    embed.add_field(name="By Alert Type", value=alert_type_text or "None", inline=True)

    # Add symbols field
    symbol_text = ""
    for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:  # Top 10
        symbol_text += f"{symbol}: {count}\n"

    embed.add_field(name="Top Symbols", value=symbol_text or "None", inline=True)

    # Add most recent alerts
    recent_text = ""
    for alert in alerts[:5]:  # Last 5 alerts
        timestamp = alert["triggered_at"].split(".")[0]  # Remove microseconds
        recent_text += f"{timestamp} - {alert['symbol']} ({alert['alert_type']})\n"

    embed.add_field(name="Recent Alerts", value=recent_text or "None", inline=False)

    await interaction.followup.send(embed=embed, ephemeral=True)


# Help command
@bot.tree.command(name="help", description="Show help information about the bot")
async def help_command(interaction: discord.Interaction):
    """Show help information"""
    await interaction.response.defer(ephemeral=True)
    
    # Create embed for help command (without pin reaction option)
    embed = discord.Embed(
        title="🚀 Crypto Trading Alerts Bot",
        description="I'll send alerts when technical indicators signal trading opportunities.",
        color=discord.Color.blue(),
    )

    embed.add_field(
        name="Getting Started",
        value="Use `/watch BTCUSDT` to start monitoring Bitcoin\n"
        "Use `/list` to see your watched pairs\n"
        "Use `/settings` to customize your alerts",
        inline=False,
    )

    embed.add_field(
        name="Available Commands",
        value="`/watch symbol [interval]` - Add a trading pair to watchlist\n"
        "`/unwatch symbol [interval]` - Remove a pair from watchlist\n"
        "`/list` - Show your watchlist\n"
        "`/settings` - View your alert settings\n"
        "`/alerts` - Enable or disable specific alert types\n"
        "`/stats` - View alert statistics\n"
        "`/guide` - Get guidance on optimal timeframes\n"
        "`/help` - Show this help information\n"
        "`/cleanup` - Delete old alert messages",
        inline=False,
    )

    embed.add_field(
        name="Alert Reactions",
        value="❓ - Get an explanation of what the alert means",
        inline=False,
    )

    embed.add_field(
        name="Credits",
        value="Made by [gabrielsaban](https://github.com/gabrielsaban)\n"
        "[View Project on GitHub](https://github.com/gabrielsaban/discord-trading-alerts)",
        inline=False,
    )

    # Send the help information directly as an ephemeral response
    await interaction.followup.send(embed=embed, ephemeral=True)


# Guide command for optimal timeframes
@bot.tree.command(
    name="guide",
    description="Get guidance on the best timeframes for different indicators",
)
@app_commands.checks.cooldown(1, 30)  # Limit usage to once every 30 seconds
async def guide_command(interaction: discord.Interaction):
    """Show a guide for optimal indicator timeframes"""
    await interaction.response.defer(ephemeral=True)

    # First embed with first set of indicators
    embed = discord.Embed(
        title="📊 Optimal Timeframes for Trading Indicators",
        description="Different indicators work best at different timeframes. Here's a guide to help you choose the right settings:",
        color=discord.Color.blue(),
    )

    # RSI Section
    embed.add_field(
        name="🔴 RSI (Relative Strength Index)",
        value="• **Short-term:** 5m, 15m, 30m - Good for scalping and quick trades\n"
        "• **Medium-term:** 1h, 4h - Most commonly used for day trading\n"
        "• **Long-term:** Daily, Weekly - For position trading and major trend reversals",
        inline=False,
    )

    # MACD Section
    embed.add_field(
        name="📈 MACD (Moving Average Convergence Divergence)",
        value="• **Medium-term:** 1h, 4h - Best for catching trend changes\n"
        "• **Long-term:** Daily - Excellent for identifying significant shifts\n"
        "• Not recommended for very short timeframes due to noise",
        inline=False,
    )

    # EMA Section
    embed.add_field(
        name="📉 EMA Crossovers",
        value="• **Short-term:** 15m, 30m - For quick trend identification\n"
        "• **Medium-term:** 1h, 4h - Most reliable for avoiding false signals\n"
        "• **Long-term:** Daily - For major trend shifts",
        inline=False,
    )

    # Bollinger Bands Section
    embed.add_field(
        name="🔄 Bollinger Bands",
        value="• **Short-term:** 15m, 30m - Good for volatility-based scalping\n"
        "• **Medium-term:** 1h, 4h - Best for identifying squeezes and breakouts\n"
        "• Works well across most timeframes as it adapts to volatility",
        inline=False,
    )

    # Second embed with remaining indicators
    embed2 = discord.Embed(color=discord.Color.blue())

    # Volume Spikes Section
    embed2.add_field(
        name="📊 Volume Spikes",
        value="• **Short-term:** 5m, 15m, 30m - Useful for catching sudden interest\n"
        "• **Medium-term:** 1h, 4h - More reliable signals with less noise\n"
        "• **Long-term:** Daily - For identifying major market events",
        inline=False,
    )

    # ADX Section
    embed2.add_field(
        name="📏 ADX (Average Directional Index)",
        value="• **Medium-term:** 1h, 4h - Ideal for trend strength measurement\n"
        "• **Long-term:** Daily - Best for identifying significant trends\n"
        "• Not recommended for very short timeframes",
        inline=False,
    )

    # Candlestick Patterns Section
    embed2.add_field(
        name="🕯️ Candlestick Patterns",
        value="• **Short-term:** 15m, 30m - Can work but more prone to false signals\n"
        "• **Medium-term:** 1h, 4h - Optimal balance of signal quality and timeliness\n"
        "• **Long-term:** Daily - Most reliable with strongest predictive value",
        inline=False,
    )

    # General Guidelines Section
    embed2.add_field(
        name="💡 General Guidelines",
        value='• The 4-hour timeframe is often the "sweet spot" for most indicators\n'
        "• Shorter timeframes generate more signals but with lower reliability\n"
        "• Longer timeframes generate fewer but more reliable signals\n"
        "• For most traders, focusing on 1h, 4h, and daily provides the best balance\n"
        "• Consider looking for confluence across multiple timeframes for the strongest signals",
        inline=False,
    )

    # Send both embeds
    await interaction.followup.send(embeds=[embed, embed2], ephemeral=True)


# Add a sync command that only the bot owner can use
@bot.tree.command(
    name="sync", description="Sync slash commands with Discord (owner only)"
)
@app_commands.describe(
    scope="Sync scope: 'global' (all servers) or 'guild' (current server only)",
    clear_commands="Delete all commands before syncing new ones",
)
async def sync_command(
    interaction: discord.Interaction,
    scope: Literal["global", "guild"] = "global",
    clear_commands: bool = False,
):
    """Sync all slash commands with Discord"""
    # Check if the user is the bot owner
    bot_owner_id = "153242761531752449"  # Your Discord ID
    if str(interaction.user.id) != bot_owner_id:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    try:
        await interaction.response.defer(ephemeral=True)

        # Filter out purge commands from app commands
        for cmd in bot.tree.get_commands():
            if cmd.name in ["purge", "purge_force"]:
                # Remove any purge commands from the command tree
                bot.tree._remove_command(cmd)
                logger.info(f"Removed command '{cmd.name}' from command tree")

        # Clear commands if requested
        if clear_commands:
            if scope == "global":
                bot.tree.clear_commands(guild=None)
                logger.info("Cleared all global commands")
            else:
                bot.tree.clear_commands(guild=interaction.guild)
                logger.info(f"Cleared all commands in guild {interaction.guild.id}")

        # Sync commands
        if scope == "global":
            # Use copy=False to ensure we're not just copying guild commands globally
            await bot.tree.sync(guild=None)
            commands = await bot.tree.fetch_commands()
            logger.info(
                f"Synced {len(commands)} global commands: {[cmd.name for cmd in commands]}"
            )

            command_list = "\n".join(
                [f"/{cmd.name} - {cmd.description}" for cmd in commands]
            )
            await interaction.followup.send(
                f"✅ Synced {len(commands)} global commands with Discord!\n\nAvailable commands:\n{command_list}",
                ephemeral=True,
            )
        else:
            # Guild-specific sync
            await bot.tree.sync(guild=interaction.guild)
            commands = await bot.tree.fetch_commands(guild=interaction.guild)
            logger.info(
                f"Synced {len(commands)} guild commands for {interaction.guild.name}: {[cmd.name for cmd in commands]}"
            )

            command_list = "\n".join(
                [f"/{cmd.name} - {cmd.description}" for cmd in commands]
            )
            await interaction.followup.send(
                f"✅ Synced {len(commands)} commands to this server only!\n\nAvailable commands:\n{command_list}",
                ephemeral=True,
            )
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
        await interaction.followup.send(
            f"❌ Error syncing commands: {str(e)}\n\nCheck logs for details.",
            ephemeral=True,
        )


# Command error handler
@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
):
    """Handle errors from slash commands"""
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"This command is on cooldown. Try again in {error.retry_after:.1f} seconds.",
            ephemeral=True,
        )
    elif isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "You don't have permission to use this command.", ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"An error occurred: {error}", ephemeral=True
        )
        logger.error(f"Command error: {error}")


# Add cleanup command
@bot.tree.command(
    name="cleanup", description="Delete old alert messages from the channel"
)
@app_commands.describe(count="Maximum number of messages to check (default: 50)")
@app_commands.checks.cooldown(1, 30)  # Limit usage to once every 30 seconds
async def cleanup_command(interaction: discord.Interaction, count: int = 50):
    """Clean up old alert messages from the channel"""
    await interaction.response.defer(ephemeral=True)

    if count < 1 or count > 100:
        await interaction.followup.send(
            "Count must be between 1 and 100.", ephemeral=True
        )
        return

    channel = interaction.channel
    user_id = str(interaction.user.id)

    # Track how many messages were deleted
    deleted_count = 0

    # Record the time when the command was called
    command_time = datetime.now(timezone.utc)

    try:
        # Only check messages in this channel
        messages_checked = 0
        async for message in channel.history(limit=count):
            messages_checked += 1

            # Skip messages created after the command was called
            if message.created_at > command_time:
                continue

            # Check if this is our alert message or batch summary
            is_bot_message = message.author.id == bot.user.id and message.embeds and len(message.embeds) > 0 and message.embeds[0].title
            is_alert = is_bot_message and "Alert:" in message.embeds[0].title
            is_summary = is_bot_message and "Batch Summary:" in message.embeds[0].title

            if is_alert or is_summary:
                try:
                    await message.delete()
                    deleted_count += 1

                    # If in our tracked messages, remove it
                    if message.id in bot.alert_messages:
                        del bot.alert_messages[message.id]

                    # Add a longer delay to avoid rate limits (Discord's rate limit is 1 per second)
                    await asyncio.sleep(1.1)
                except Exception as e:
                    logger.error(f"Error deleting message during cleanup: {e}")

        # Send confirmation
        await interaction.followup.send(
            f"✅ Cleanup complete! Checked {messages_checked} messages and deleted {deleted_count} alert and summary messages.",
            ephemeral=True,
        )
    except Exception as e:
        logger.error(f"Error during cleanup command: {e}")
        await interaction.followup.send(
            f"❌ Error during cleanup: {str(e)}", ephemeral=True
        )


# Run the bot
def run_bot():
    """Run the Discord bot"""
    try:
        bot.run(TOKEN)
    except discord.errors.LoginFailure:
        logger.error("Invalid token provided. Please check your token and try again.")
    except Exception as e:
        logger.error(f"Error running bot: {e}")


if __name__ == "__main__":
    run_bot()
