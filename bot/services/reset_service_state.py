"""Utility module for resetting service state during testing"""
import importlib
import logging

logger = logging.getLogger("discord_trading_alerts.test_utils")


def reset_all_service_singletons():
    """
    Reset all singleton instances for clean testing.
    This function should be called before each test that needs a clean state.
    """
    # Reset CooldownService singleton
    import bot.services.cooldown_service

    bot.services.cooldown_service._cooldown_service = None

    # Reset OverrideEngine singleton
    import bot.services.override_engine

    bot.services.override_engine._override_engine_instance = None

    # Reset BatchAggregator singleton
    import bot.services.batch_aggregator

    bot.services.batch_aggregator._batch_aggregator = None

    # Reset CooldownRepository singleton if it exists
    try:
        import bot.services.cooldown_repository

        if hasattr(bot.services.cooldown_repository.CooldownRepository, "_instance"):
            bot.services.cooldown_repository.CooldownRepository._instance = None
    except (ImportError, AttributeError):
        logger.debug("Could not reset CooldownRepository singleton")

    # Reset AlertManager class-level cooldowns
    from bot.alerts import AlertManager

    AlertManager.global_cooldowns = {}

    logger.debug("All service singletons have been reset")
