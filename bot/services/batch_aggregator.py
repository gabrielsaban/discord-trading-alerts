import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import feature flags
from bot.services.feature_flags import get_flag

logger = logging.getLogger("discord_trading_alerts.batch_aggregator")


class BatchAggregator:
    """
    Service for batching and aggregating alerts that would otherwise be blocked by cooldowns.
    Buffers alerts for 5 minutes, then groups and emits the top-2 strongest alerts per symbol.
    """

    def __init__(self):
        """Initialize the batch aggregator with default settings"""
        # Dictionary to store queued alerts by user_id -> symbol -> alert_type -> list of alerts
        self.queued_alerts = {}

        # Lock for thread-safe access to queued_alerts
        self.lock = threading.RLock()

        # Batch processing interval (5 minutes)
        self.batch_interval = 5  # minutes

        # Track the last processing time per user
        self.last_processed = {}

        # Track alert count since last processing to force processing when enough alerts accumulate
        self.alert_count_since_last = {}

        # Threshold to trigger processing regardless of timer (if 10+ alerts are queued)
        self.force_process_threshold = 10

        # Callback function to send alerts
        self.callback = None

        # Flag to track if the background task is running
        self.background_task_running = False

        # Event loop for background task
        self.loop = None

        # Background task
        self.background_task = None

        logger.info("BatchAggregator initialized")

    def set_callback(self, callback: Callable[[str, List[str]], None]) -> None:
        """
        Set the callback function to send alerts.

        Parameters:
        -----------
        callback : Callable[[str, List[str]], None]
            Function that takes a user_id and a list of alert messages
        """
        self.callback = callback
        logger.debug(f"Alert callback set: {callback}")

    def enqueue(
        self,
        user_id: str,
        symbol: str,
        interval: str,
        alert_type: str,
        alert_subtype: str,
        alert_msg: str,
        strength: float,
    ) -> None:
        """
        Enqueue an alert that would be blocked by cooldown/override.

        Parameters:
        -----------
        user_id : str
            ID of the user to send the alert to
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Timeframe of the alert (e.g., '5m', '1h')
        alert_type : str
            Type of alert (e.g., 'RSI', 'MACD')
        alert_subtype : str
            Subtype of alert (e.g., 'OVERSOLD', 'BULLISH CROSS')
        alert_msg : str
            Full alert message
        strength : float
            Signal strength (1.0-10.0)
        """
        # Check if batch aggregator is enabled via feature flag
        if not get_flag("ENABLE_BATCH_AGGREGATOR", False):
            logger.debug("BatchAggregator disabled by feature flag")
            return

        with self.lock:
            # Initialize dictionaries if they don't exist
            if user_id not in self.queued_alerts:
                self.queued_alerts[user_id] = {}
                self.alert_count_since_last[user_id] = 0

            if symbol not in self.queued_alerts[user_id]:
                self.queued_alerts[user_id][symbol] = {}

            # Create a key that combines alert_type and subtype
            alert_key = f"{alert_type}_{alert_subtype or 'general'}"

            if alert_key not in self.queued_alerts[user_id][symbol]:
                self.queued_alerts[user_id][symbol][alert_key] = []

            # Add the alert to the queue
            timestamp = datetime.utcnow()
            self.queued_alerts[user_id][symbol][alert_key].append(
                {
                    "message": alert_msg,
                    "timestamp": timestamp,
                    "interval": interval,
                    "strength": strength,
                    "type": alert_type,
                    "subtype": alert_subtype,
                }
            )

            # Increment the alert count
            self.alert_count_since_last[user_id] = (
                self.alert_count_since_last.get(user_id, 0) + 1
            )

            logger.info(
                f"Enqueued alert for {user_id}: {symbol} ({interval}) - {alert_key}"
            )

            # Start background task if not already running
            if not self.background_task_running:
                self.start_background_task()

    def start_background_task(self) -> None:
        """Start the background task to process batched alerts"""
        if self.background_task_running:
            logger.debug("Background task already running")
            return

        try:
            # Use the current event loop or create a new one
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

            # Start the background task
            self.background_task = self.loop.create_task(
                self._process_batches_periodically()
            )
            self.background_task_running = True
            logger.info("Started background task for processing batched alerts")
        except Exception as e:
            logger.error(f"Error starting background task: {e}")

    async def _process_batches_periodically(self) -> None:
        """Background task to process batched alerts every 5 minutes"""
        try:
            while True:
                # Check if batch aggregator is still enabled
                if not get_flag("ENABLE_BATCH_AGGREGATOR", False):
                    logger.info(
                        "BatchAggregator disabled by feature flag, stopping background task"
                    )
                    self.background_task_running = False
                    break

                # Verify callback is set before processing
                if self.callback is None:
                    logger.error(
                        "BatchAggregator callback is not set! Cannot process batches."
                    )
                    logger.error(
                        "Please ensure batch_aggregator.set_callback() is called during initialization."
                    )

                    # Don't process if callback isn't set
                    await asyncio.sleep(60)  # Wait a minute and try again
                    continue

                # Process batches for all users
                try:
                    logger.info("Processing batched alerts (background task)")
                    self._process_all_batches()
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

                # Wait for the batch interval
                await asyncio.sleep(self.batch_interval * 60)
        except asyncio.CancelledError:
            logger.info("Background task cancelled")
            self.background_task_running = False
        except Exception as e:
            logger.error(f"Error in background task: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.background_task_running = False

    def _process_all_batches(self) -> None:
        """Process batched alerts for all users"""
        with self.lock:
            now = datetime.utcnow()

            # Add debug log to show processing started
            logger.info(
                f"Processing batches at {now.strftime('%H:%M:%S')} for {len(self.queued_alerts)} users"
            )

            # Process for each user
            for user_id in list(self.queued_alerts.keys()):
                # Skip if no alerts or last processed time is too recent
                if not self.queued_alerts[user_id]:
                    logger.debug(f"Skipping user {user_id} - no queued alerts")
                    continue

                # Add debug log to show total alerts queued for this user
                total_alerts = sum(
                    len(alerts)
                    for symbol_alerts in self.queued_alerts[user_id].values()
                    for alerts in symbol_alerts.values()
                )
                logger.debug(
                    f"User {user_id} has {total_alerts} total alerts queued across {len(self.queued_alerts[user_id])} symbols"
                )

                # Get time since last processing
                last_time = self.last_processed.get(user_id, datetime.min)
                time_since_last = (now - last_time).total_seconds() / 60

                # Process if enough time has passed OR if we have accumulated enough alerts
                force_by_count = (
                    self.alert_count_since_last.get(user_id, 0)
                    >= self.force_process_threshold
                )
                should_process = (
                    time_since_last >= self.batch_interval or force_by_count
                )

                if not should_process:
                    logger.debug(
                        f"Skipping batch processing for {user_id}, last processed {time_since_last:.1f} minutes ago (< {self.batch_interval})"
                    )
                    continue

                # Process batches for this user
                self._process_user_batches(user_id)

                # Update last processed time and reset counter
                self.last_processed[user_id] = now
                self.alert_count_since_last[user_id] = 0

    def _process_user_batches(self, user_id: str) -> None:
        """
        Process batched alerts for a specific user

        Parameters:
        -----------
        user_id : str
            ID of the user to process alerts for
        """
        if not self.callback:
            logger.error(
                "No callback function set, cannot process batches. Please ensure batch_aggregator.set_callback() is called during initialization."
            )
            return

        # Check if user has any alerts
        if user_id not in self.queued_alerts or not self.queued_alerts[user_id]:
            logger.debug(f"No queued alerts for user {user_id}")
            return

        alerts_to_send = []

        # Process each symbol for this user
        for symbol, alert_types in self.queued_alerts[user_id].items():
            # Skip if no alerts for this symbol
            if not alert_types:
                continue

            # Log the alert types and count for this symbol
            alert_counts = {key: len(alerts) for key, alerts in alert_types.items()}
            logger.debug(
                f"Processing {sum(alert_counts.values())} alerts for {symbol}: {alert_counts}"
            )

            symbol_results = self._process_symbol_batches(symbol, alert_types)
            alerts_to_send.extend(symbol_results)

        # Send the processed alerts if any
        if alerts_to_send:
            try:
                logger.info(
                    f"Calling callback with {len(alerts_to_send)} alerts for {user_id}"
                )
                self.callback(user_id, alerts_to_send)
                logger.info(f"Sent {len(alerts_to_send)} batched alerts to {user_id}")
            except Exception as e:
                logger.error(f"Error sending batched alerts to {user_id}: {e}")
                logger.error(f"Callback: {self.callback}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.debug(f"No alerts to send for user {user_id} after processing")

    def _process_symbol_batches(self, symbol: str, alert_types: Dict) -> List[str]:
        """
        Process batched alerts for a specific symbol

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_types : Dict
            Dictionary of alert types and their queued alerts

        Returns:
        --------
        List[str]
            List of processed alert messages to send
        """
        if not alert_types:
            return []

        result = []

        # Find top alerts across all alert types
        all_alerts = []
        for alert_key, alerts in alert_types.items():
            all_alerts.extend(alerts)

        # If no alerts, return empty list
        if not all_alerts:
            return []

        # Group alerts by type
        grouped_alerts = {}
        for alert in all_alerts:
            alert_type = alert["type"]
            if alert_type not in grouped_alerts:
                grouped_alerts[alert_type] = []
            grouped_alerts[alert_type].append(alert)

        # Sort each group by strength and select top 2 for each type
        top_alerts = []
        for alert_type, alerts in grouped_alerts.items():
            # Sort by strength (descending)
            sorted_alerts = sorted(
                alerts, key=lambda x: x.get("strength", 0), reverse=True
            )
            # Take up to 2 top alerts
            top_alerts.extend(sorted_alerts[:2])

        # Handle special case for single alert
        if len(top_alerts) == 1:
            result.append(top_alerts[0]["message"])

        # Handle multiple alerts - create a summary
        elif len(top_alerts) > 1:
            # Count total alerts for title
            total_alert_count = len(all_alerts)

            # Group count by alert type for summary line
            type_counts = {}
            for alert in top_alerts:
                alert_type = alert["type"]
                if alert_type not in type_counts:
                    type_counts[alert_type] = 0
                type_counts[alert_type] += 1

            # Build summary message with total count in title
            summary = (
                f"🔔 **ALERT SUMMARY** for {symbol} ({total_alert_count} alerts)\n\n"
            )

            # Add counts by type
            type_lines = []
            for alert_type, count in type_counts.items():
                type_lines.append(
                    f"{count} {alert_type} alert{'s' if count > 1 else ''}"
                )

            summary += "Including: " + ", ".join(type_lines) + "\n\n"

            # Improved grouping for better consolidation
            grouped_by_message = {}
            for alert in top_alerts:
                # Create a better key that captures the essence of the alert
                message_lines = alert["message"].split("\n")
                header = message_lines[0]

                # Extract the core alert message without the emoji prefix
                if ":" in header:
                    alert_core = header.split(":", 1)[1].strip()
                else:
                    alert_core = header

                # Create key using alert core message to better group similar alerts
                # We'll combine alerts with the same pattern and interval
                key = f"{alert_core}|{alert['interval']}"

                if key not in grouped_by_message:
                    grouped_by_message[key] = {
                        "alerts": [],
                        "intervals": set(),
                        "max_strength": 0,
                        "earliest_time": alert["timestamp"],
                        "prices": set(),
                        "additional_data": {},
                        "emoji": header.split(" ")[0] if " " in header else "🔔",
                    }

                # Add this alert to the group
                grouped_by_message[key]["alerts"].append(alert)
                grouped_by_message[key]["intervals"].add(alert["interval"] or "unknown")
                grouped_by_message[key]["max_strength"] = max(
                    grouped_by_message[key]["max_strength"], alert.get("strength", 5.0)
                )

                # Track the earliest timestamp
                if alert["timestamp"] < grouped_by_message[key]["earliest_time"]:
                    grouped_by_message[key]["earliest_time"] = alert["timestamp"]

                # Collect all price data
                for line in message_lines:
                    if "Price:" in line:
                        price_value = line.split("Price:")[1].strip()
                        grouped_by_message[key]["prices"].add(price_value)

                    # Collect any other data points (Bandwidth, RSI, etc.)
                    elif (
                        ":" in line and not line.startswith("**") and not "http" in line
                    ):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            data_key = parts[0].strip()
                            data_value = parts[1].strip()

                            if (
                                data_key
                                not in grouped_by_message[key]["additional_data"]
                            ):
                                grouped_by_message[key]["additional_data"][
                                    data_key
                                ] = set()

                            grouped_by_message[key]["additional_data"][data_key].add(
                                data_value
                            )

            # Sort grouped alerts by max strength
            sorted_groups = sorted(
                grouped_by_message.values(),
                key=lambda x: x["max_strength"],
                reverse=True,
            )

            # Show top grouped alerts with improved formatting
            for i, group in enumerate(sorted_groups[:5], 1):
                # Use representative alert for core info
                alert = group["alerts"][0]

                # Extract alert header without emoji prefix
                message_lines = alert["message"].split("\n")
                header = message_lines[0]

                if ":" in header:
                    alert_title = header.split(":", 1)[1].strip()
                else:
                    alert_title = header

                # Format intervals
                intervals = "/".join(sorted(group["intervals"]))

                # Format time
                time_str = group["earliest_time"].strftime("%H:%M:%S")

                # Get max strength
                strength = group["max_strength"]

                # Count prefix for multiple alerts
                count_prefix = (
                    f"({len(group['alerts'])}x) " if len(group["alerts"]) > 1 else ""
                )

                # Add emoji from original alert
                emoji = group["emoji"]

                # Format the main summary line
                summary += f"{i}. {count_prefix}{intervals} @ {time_str} (strength: {strength:.1f}): {emoji} {alert_title}\n"

                # Add price information (joined if multiple values)
                if group["prices"]:
                    price_list = sorted(group["prices"])
                    if len(price_list) == 1:
                        summary += f"   Price: {price_list[0]}\n"
                    else:
                        # Show range or list of prices
                        summary += f"   Prices: {', '.join(price_list)}\n"

                # Add any additional data points
                for data_key, data_values in sorted(group["additional_data"].items()):
                    if data_values:
                        data_list = sorted(data_values)
                        if len(data_list) == 1:
                            summary += f"   {data_key}: {data_list[0]}\n"
                        else:
                            # Show range or list of values
                            summary += f"   {data_key}: {', '.join(data_list)}\n"

                summary += "\n"

            # Show count of remaining alerts
            remaining = total_alert_count - sum(
                len(group["alerts"]) for group in sorted_groups[:5]
            )
            if remaining > 0:
                summary += f"_{remaining} more signals not shown_\n"

            result.append(summary)

        # Clear the processed alerts
        for alert_key in list(alert_types.keys()):
            alert_types[alert_key] = []

        return result

    def clear_all(self) -> None:
        """Clear all queued alerts"""
        with self.lock:
            self.queued_alerts.clear()
            self.last_processed.clear()
            self.alert_count_since_last.clear()
            logger.info("Cleared all queued alerts")

    def stop(self) -> None:
        """Stop the background task"""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            logger.info("Background task cancelled")

        self.background_task_running = False


# Singleton instance
_batch_aggregator = None


def get_batch_aggregator() -> BatchAggregator:
    """Get the singleton BatchAggregator instance"""
    global _batch_aggregator
    if _batch_aggregator is None:
        _batch_aggregator = BatchAggregator()
    return _batch_aggregator


def check_batch_aggregator_callback() -> bool:
    """Check if the BatchAggregator callback is properly set

    Returns:
    --------
    bool
        True if callback is set, False if not
    """
    aggregator = get_batch_aggregator()
    if aggregator.callback is None:
        logger.error("❌ BatchAggregator callback is NOT SET. Alerts will NOT be sent!")
        return False
    else:
        logger.info(f"✅ BatchAggregator callback is set: {aggregator.callback}")
        return True
