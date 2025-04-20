import os
import discord
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Literal
from discord import app_commands
from dotenv import load_dotenv

# Load our modules
from bot.db import get_db
from bot.scheduler import get_scheduler
from bot.binance import fetch_market_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_alerts.discord_bot')

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
if not TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in .env file")
    raise ValueError("DISCORD_BOT_TOKEN environment variable is required")

# Valid intervals
VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']


# Set up intents
intents = discord.Intents.default()
intents.message_content = True  # Allow the bot to see message content

# Define the bot client
class TradingAlertsBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.db = get_db()
        self.alert_channels: Dict[str, List[discord.TextChannel]] = {}  # user_id -> list of channels
        self.synced = False
        self.scheduler = None
    
    async def setup_hook(self):
        """Set up the bot's background tasks and sync commands"""
        # Start the scheduler with our notification callback
        self.scheduler = get_scheduler(self.send_alert_notification)
        self.scheduler.initialize()
        
        # Sync commands with Discord
        await self.tree.sync()
        self.synced = True
        logger.info("Command tree synced with Discord")
    
    async def on_ready(self):
        """Called when the bot has connected to Discord"""
        logger.info(f'{self.user} has connected to Discord!')
        
        # Set up status message
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="/watch for trading alerts"
            )
        )
    
    async def on_guild_join(self, guild):
        """Called when the bot joins a new guild (server)"""
        logger.info(f"Bot joined guild: {guild.name} (ID: {guild.id})")
        
        # Try to find a general or bot channel to send welcome message
        channel = None
        for ch in guild.text_channels:
            if ch.permissions_for(guild.me).send_messages:
                if ch.name.lower() in ['general', 'bot', 'bot-commands', 'commands', 'alerts']:
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
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Getting Started",
            value="Use `/watch BTCUSDT` to start monitoring Bitcoin\n"
                  "Use `/list` to see your watched pairs\n"
                  "Use `/settings` to customize your alerts",
            inline=False
        )
        
        embed.add_field(
            name="Available Commands",
            value="`/watch symbol [interval]` - Add a trading pair to watchlist\n"
                  "`/unwatch symbol [interval]` - Remove a pair from watchlist\n"
                  "`/list` - Show your watchlist\n"
                  "`/settings` - Adjust alert thresholds\n"
                  "`/stats` - View alert statistics",
            inline=False
        )
        
        embed.add_field(
            name="Credits",
            value="Made by [gabrielsaban](https://github.com/gabrielsaban)\n"
                  "[View Project on GitHub](https://github.com/gabrielsaban/discord-trading-alerts)",
            inline=False
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
        
        # Send alerts to all registered channels
        for channel in self.alert_channels[user_id]:
            try:
                for alert in alerts:
                    # Extract symbol from alert message
                    symbol = alert.split(':')[1].split()[0] if ':' in alert else "Unknown"
                    
                    # Create embed
                    embed = discord.Embed(
                        title=f"⚠️ Alert: {symbol}",
                        description=alert,
                        color=self._get_color_for_alert(alert)
                    )
                    
                    # Add timestamp
                    embed.timestamp = discord.utils.utcnow()
                    
                    # Send the embed
                    await channel.send(embed=embed)
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error sending alert to channel {channel.id}: {e}")
    
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
    
    def register_channel(self, user_id: str, channel: discord.TextChannel):
        """Register a channel for a user's alerts"""
        if user_id not in self.alert_channels:
            self.alert_channels[user_id] = []
        
        if channel not in self.alert_channels[user_id]:
            self.alert_channels[user_id].append(channel)
            logger.info(f"Registered channel {channel.id} for user {user_id}")
    
    def unregister_channel(self, user_id: str, channel: discord.TextChannel):
        """Unregister a channel for a user's alerts"""
        if user_id in self.alert_channels and channel in self.alert_channels[user_id]:
            self.alert_channels[user_id].remove(channel)
            logger.info(f"Unregistered channel {channel.id} for user {user_id}")


# Create bot instance
bot = TradingAlertsBot()


# Watch command
@bot.tree.command(
    name="watch",
    description="Add a trading pair to your watchlist"
)
@app_commands.describe(
    symbol="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    interval="Time interval for candles (default: 15m)",
    channel="Channel to send alerts to (default: current channel)"
)
@app_commands.choices(interval=[
    app_commands.Choice(name=interval, value=interval)
    for interval in VALID_INTERVALS
])
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def watch_command(
    interaction: discord.Interaction, 
    symbol: str, 
    interval: str = "15m",
    channel: Optional[discord.TextChannel] = None
):
    """Add a trading pair to the user's watchlist"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = str(interaction.user.id)
    symbol = symbol.upper()  # Standardize to uppercase
    
    # Validate interval
    if interval not in VALID_INTERVALS:
        await interaction.followup.send(
            f"Invalid interval: {interval}. Valid intervals are: {', '.join(VALID_INTERVALS)}",
            ephemeral=True
        )
        return
    
    # Validate the symbol by attempting to fetch data
    try:
        df = fetch_market_data(symbol=symbol, interval=interval, limit=5)
        if df is None or df.empty:
            await interaction.followup.send(
                f"Could not fetch data for {symbol}. Please check that the symbol is valid on Binance.",
                ephemeral=True
            )
            return
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {e}")
        await interaction.followup.send(
            f"Error validating {symbol}: {str(e)}. Please check that the symbol is valid on Binance.",
            ephemeral=True
        )
        return
    
    # Get or create user
    user = bot.db.get_user(user_id)
    if not user:
        bot.db.create_user(user_id, interaction.user.display_name, interaction.user.id)
    
    # Add to watchlist
    success = bot.db.add_to_watchlist(user_id, symbol, interval)
    
    if success:
        # Register the channel for alerts
        alert_channel = channel or interaction.channel
        bot.register_channel(user_id, alert_channel)
        
        # Add to scheduler
        bot.scheduler.add_symbol(symbol, interval)
        
        # Notify user
        await interaction.followup.send(
            f"🔔 Added {symbol} ({interval}) to your watchlist. Alerts will be sent to {alert_channel.mention}.",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"Failed to add {symbol} to your watchlist. It may already be in your watchlist.",
            ephemeral=True
        )


# Unwatch command
@bot.tree.command(
    name="unwatch",
    description="Remove a trading pair from your watchlist"
)
@app_commands.describe(
    symbol="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    interval="Time interval for candles (default: 15m)"
)
@app_commands.choices(interval=[
    app_commands.Choice(name=interval, value=interval)
    for interval in VALID_INTERVALS
])
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def unwatch_command(
    interaction: discord.Interaction, 
    symbol: str, 
    interval: str = "15m"
):
    """Remove a trading pair from the user's watchlist"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = str(interaction.user.id)
    symbol = symbol.upper()  # Standardize to uppercase
    
    # Validate interval
    if interval not in VALID_INTERVALS:
        await interaction.followup.send(
            f"Invalid interval: {interval}. Valid intervals are: {', '.join(VALID_INTERVALS)}",
            ephemeral=True
        )
        return
    
    # Remove from watchlist
    success = bot.db.remove_from_watchlist(user_id, symbol, interval)
    
    if success:
        await interaction.followup.send(
            f"🔕 Removed {symbol} ({interval}) from your watchlist.",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"{symbol} ({interval}) is not in your watchlist.",
            ephemeral=True
        )


# List command
@bot.tree.command(
    name="list",
    description="Show your watched trading pairs"
)
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
            ephemeral=True
        )
        return
    
    # Create embed
    embed = discord.Embed(
        title="🔍 Your Watchlist",
        description=f"You are watching {len(watchlist)} trading pairs:",
        color=discord.Color.blue()
    )
    
    # Group by interval
    intervals = {}
    for item in watchlist:
        interval = item['interval']
        if interval not in intervals:
            intervals[interval] = []
        intervals[interval].append(item['symbol'])
    
    # Add fields for each interval
    for interval, symbols in intervals.items():
        embed.add_field(
            name=f"{interval} Interval",
            value=", ".join(symbols) or "None",
            inline=False
        )
    
    await interaction.followup.send(embed=embed, ephemeral=True)


# Settings command
@bot.tree.command(
    name="settings",
    description="Adjust your alert settings"
)
@app_commands.describe(
    setting="Setting to change",
    value="New value for the setting"
)
@app_commands.choices(setting=[
    app_commands.Choice(name="RSI Oversold Threshold", value="rsi_oversold"),
    app_commands.Choice(name="RSI Overbought Threshold", value="rsi_overbought"),
    app_commands.Choice(name="Volume Spike Threshold", value="volume_threshold"),
    app_commands.Choice(name="EMA Short Period", value="ema_short"),
    app_commands.Choice(name="EMA Long Period", value="ema_long"),
    app_commands.Choice(name="Bollinger Band Squeeze Threshold", value="bb_squeeze_threshold"),
    app_commands.Choice(name="ADX Trend Strength Threshold", value="adx_threshold"),
    app_commands.Choice(name="Alert Cooldown (minutes)", value="cooldown_minutes")
])
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def settings_command(
    interaction: discord.Interaction,
    setting: Optional[str] = None,
    value: Optional[str] = None
):
    """View or adjust alert settings"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = str(interaction.user.id)
    
    # Get user settings
    user = bot.db.get_user(user_id)
    if not user:
        bot.db.create_user(user_id, interaction.user.display_name, interaction.user.id)
        user = bot.db.get_user(user_id)
    
    # If no setting provided, show current settings
    if setting is None:
        embed = discord.Embed(
            title="⚙️ Your Alert Settings",
            description="These settings apply to all your alerts. Use `/settings setting value` to change them.",
            color=discord.Color.blue()
        )
        
        settings = user['settings']
        
        embed.add_field(
            name="RSI Thresholds",
            value=f"Oversold: {settings.get('rsi_oversold', 30)}\n"
                  f"Overbought: {settings.get('rsi_overbought', 70)}",
            inline=True
        )
        
        embed.add_field(
            name="EMA Periods",
            value=f"Short: {settings.get('ema_short', 9)}\n"
                  f"Long: {settings.get('ema_long', 21)}",
            inline=True
        )
        
        embed.add_field(
            name="Volume & Bollinger",
            value=f"Volume Threshold: {settings.get('volume_threshold', 2.5)}\n"
                  f"BB Squeeze: {settings.get('bb_squeeze_threshold', 0.05)}",
            inline=True
        )
        
        embed.add_field(
            name="Trend & Timing",
            value=f"ADX Threshold: {settings.get('adx_threshold', 25)}\n"
                  f"Cooldown: {settings.get('cooldown_minutes', 240)} minutes",
            inline=True
        )
        
        # Show enabled alert types
        enabled = settings.get('enabled_alerts', ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"])
        embed.add_field(
            name="Enabled Alerts",
            value=", ".join(enabled) or "None",
            inline=False
        )
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        return
    
    # If setting provided but no value, show current value
    if value is None:
        current_value = user['settings'].get(setting, "Not set")
        await interaction.followup.send(
            f"Current value for {setting}: {current_value}\n"
            f"Use `/settings {setting} new_value` to change it.",
            ephemeral=True
        )
        return
    
    # Update setting
    try:
        # Convert value to appropriate type
        if setting in ['rsi_oversold', 'rsi_overbought', 'ema_short', 'ema_long', 'adx_threshold', 'cooldown_minutes']:
            typed_value = int(value)
        elif setting in ['volume_threshold', 'bb_squeeze_threshold']:
            typed_value = float(value)
        else:
            typed_value = value
        
        # Validate values
        if setting == 'rsi_oversold' and (typed_value < 1 or typed_value > 40):
            await interaction.followup.send(
                f"Invalid RSI oversold threshold: {typed_value}. Must be between 1 and 40.",
                ephemeral=True
            )
            return
        
        if setting == 'rsi_overbought' and (typed_value < 60 or typed_value > 99):
            await interaction.followup.send(
                f"Invalid RSI overbought threshold: {typed_value}. Must be between 60 and 99.",
                ephemeral=True
            )
            return
        
        if setting in ['ema_short', 'ema_long'] and (typed_value < 2 or typed_value > 200):
            await interaction.followup.send(
                f"Invalid EMA period: {typed_value}. Must be between 2 and 200.",
                ephemeral=True
            )
            return
        
        if setting == 'volume_threshold' and (typed_value < 1.1 or typed_value > 10):
            await interaction.followup.send(
                f"Invalid volume threshold: {typed_value}. Must be between 1.1 and 10.",
                ephemeral=True
            )
            return
        
        if setting == 'bb_squeeze_threshold' and (typed_value < 0.01 or typed_value > 0.5):
            await interaction.followup.send(
                f"Invalid Bollinger Band squeeze threshold: {typed_value}. Must be between 0.01 and 0.5.",
                ephemeral=True
            )
            return
        
        if setting == 'adx_threshold' and (typed_value < 10 or typed_value > 50):
            await interaction.followup.send(
                f"Invalid ADX threshold: {typed_value}. Must be between 10 and 50.",
                ephemeral=True
            )
            return
        
        if setting == 'cooldown_minutes' and (typed_value < 5 or typed_value > 1440):
            await interaction.followup.send(
                f"Invalid cooldown: {typed_value}. Must be between 5 and 1440 minutes (24 hours).",
                ephemeral=True
            )
            return
        
        # Update setting
        bot.db.update_user_settings(user_id, {setting: typed_value})
        
        await interaction.followup.send(
            f"✅ Updated {setting} to {typed_value}.",
            ephemeral=True
        )
    except ValueError:
        await interaction.followup.send(
            f"Invalid value: {value}. Please provide a valid number.",
            ephemeral=True
        )
    except Exception as e:
        logger.error(f"Error updating setting {setting} for user {user_id}: {e}")
        await interaction.followup.send(
            f"Error updating setting: {str(e)}",
            ephemeral=True
        )


# Enable/disable alerts command
@bot.tree.command(
    name="alerts",
    description="Enable or disable specific alert types"
)
@app_commands.describe(
    action="Enable or disable alerts",
    alert_type="Type of alert to enable/disable"
)
@app_commands.choices(
    action=[
        app_commands.Choice(name="Enable", value="enable"),
        app_commands.Choice(name="Disable", value="disable")
    ],
    alert_type=[
        app_commands.Choice(name="RSI (Relative Strength Index)", value="rsi"),
        app_commands.Choice(name="MACD (Moving Average Convergence Divergence)", value="macd"),
        app_commands.Choice(name="EMA Crossovers", value="ema"),
        app_commands.Choice(name="Bollinger Bands", value="bb"),
        app_commands.Choice(name="Volume Spikes", value="volume"),
        app_commands.Choice(name="ADX (Average Directional Index)", value="adx"),
        app_commands.Choice(name="Candlestick Patterns", value="pattern"),
        app_commands.Choice(name="All Alerts", value="all")
    ]
)
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def alerts_command(
    interaction: discord.Interaction,
    action: str,
    alert_type: str
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
    settings = user['settings']
    enabled_alerts = settings.get('enabled_alerts', ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"])
    
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
    bot.db.update_user_settings(user_id, {'enabled_alerts': enabled_alerts})
    
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
        ephemeral=True
    )


# Stats command
@bot.tree.command(
    name="stats",
    description="View statistics about your alerts"
)
@app_commands.describe(
    days="Number of days to include in the stats (default: 7)"
)
@app_commands.checks.cooldown(1, 5)  # 1 use every 5 seconds per user
async def stats_command(
    interaction: discord.Interaction,
    days: int = 7
):
    """View alert statistics"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = str(interaction.user.id)
    
    # Validate days
    if days < 1 or days > 30:
        await interaction.followup.send(
            f"Invalid number of days: {days}. Must be between 1 and 30.",
            ephemeral=True
        )
        return
    
    # Get recent alerts
    alerts = bot.db.get_recent_alerts(user_id, hours=days * 24)
    
    if not alerts:
        await interaction.followup.send(
            f"No alerts in the past {days} days. Add trading pairs with `/watch` to get started!",
            ephemeral=True
        )
        return
    
    # Create stats embed
    embed = discord.Embed(
        title=f"📊 Your Alert Stats (Past {days} days)",
        description=f"You've received {len(alerts)} alerts:",
        color=discord.Color.blue()
    )
    
    # Count by type
    alert_types = {}
    symbols = {}
    
    for alert in alerts:
        # Count by type
        alert_type = alert['alert_type']
        if alert_type not in alert_types:
            alert_types[alert_type] = 0
        alert_types[alert_type] += 1
        
        # Count by symbol
        symbol = alert['symbol']
        if symbol not in symbols:
            symbols[symbol] = 0
        symbols[symbol] += 1
    
    # Add alert types field
    alert_type_text = ""
    for alert_type, count in sorted(alert_types.items(), key=lambda x: x[1], reverse=True):
        alert_type_text += f"{alert_type.upper()}: {count}\n"
    
    embed.add_field(
        name="By Alert Type",
        value=alert_type_text or "None",
        inline=True
    )
    
    # Add symbols field
    symbol_text = ""
    for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
        symbol_text += f"{symbol}: {count}\n"
    
    embed.add_field(
        name="Top Symbols",
        value=symbol_text or "None",
        inline=True
    )
    
    # Add most recent alerts
    recent_text = ""
    for alert in alerts[:5]:  # Last 5 alerts
        timestamp = alert['triggered_at'].split('.')[0]  # Remove microseconds
        recent_text += f"{timestamp} - {alert['symbol']} ({alert['alert_type']})\n"
    
    embed.add_field(
        name="Recent Alerts",
        value=recent_text or "None",
        inline=False
    )
    
    await interaction.followup.send(embed=embed, ephemeral=True)


# Help command
@bot.tree.command(
    name="help",
    description="Show help information about the bot"
)
async def help_command(interaction: discord.Interaction):
    """Show help information"""
    await bot.send_welcome_message(interaction.channel)
    await interaction.response.send_message("Help information sent to the channel!", ephemeral=True)


# Command error handler
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Handle errors from application commands"""
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"This command is on cooldown. Try again in {error.retry_after:.1f} seconds.",
            ephemeral=True
        )
    elif isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "You don't have permission to use this command.",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"An error occurred: {error}",
            ephemeral=True
        )
        logger.error(f"Command error: {error}")


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
