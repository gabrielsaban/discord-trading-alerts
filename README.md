# 💸 Discord Crypto Trading Alerts Bot

A live Discord bot that tracks crypto pairs using real-time Binance data and notifies users of key technical indicator signals — like RSI oversold, MACD crossovers, or EMA breakouts. Clean design, scalable alerts, and easy setup.

---

## 🔧 Features

- 🔍 Monitor any **Binance-supported crypto pair** (e.g. BTCUSDT, ETHUSDT)
- 📊 Get alerts based on popular **technical indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA crossovers (e.g. 9 EMA > 21 EMA)
  - Bollinger Band breakouts
  - Volume spikes
- ⏰ **Scheduled scans** (e.g. every 10 minutes)
- 🧠 Custom **watchlists** per user
- ✅ Discord notifications via embeds or DMs
- 📂 Data stored with `SQLite` for persistence
- ☁️ Deploy-ready (Render)

---

## 🛠 Tech Stack

| Component | Tool |
|----------|------|
| Bot Framework | [discord.py](https://discordpy.readthedocs.io/) |
| Indicators | [pandas-ta](https://github.com/twopirllc/pandas-ta) |
| Data API | [Binance REST API](https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data) |
| Storage | SQLite3 |
| Scheduler | APScheduler |
| Hosting | Render |

---

## 🧊 Smart Cooling System

The bot uses an advanced alert cooling system to prevent alert spam while ensuring you don't miss important signals:

## Signal Strength Prioritization
Strong signals can override cooldowns when they're significantly more important:
- Extreme indicator readings (e.g., RSI below 20, ADX above 40)
- Higher timeframe signals get priority
- Signals with high strength can break through even if cooldown period hasn't fully elapsed

1. **ATR-Based Dynamic Cooldowns**
   - 5m: 20 min base cooldown (±5 min based on 30-min ATR)
   - 15m: 1h base cooldown (±15 min based on 1h ATR)
   - 1h: 2h base cooldown (±30 min based on 4h ATR)
   - 4h: 24h base cooldown (fixed)

2. **Market Volatility Adjustments**
   - ATR percentile mapping: Top 25% → +25% cooldown, Bottom 25% → -25% cooldown
   - High-volatility session handling: +10% cooldown on 5m and 15m during London/NY overlap

3. **Enhanced Override Logic**
   - 4h/1d signals bypass all lower-tf cooldowns, then enter their own 24h duplicate-block cooldown
   - Extreme readings override active cooldowns on same timeframe (RSI < 20 or > 80, ADX > 40)
   - Medium signals can override shorter timeframe cooldowns if strength metric exceeds threshold

---

## ⚙️ Commands

| Command | Description |
|--------|-------------|
| `/watch BTCUSDT` | Start tracking a pair |
| `/unwatch BTCUSDT` | Stop tracking |
| `/list` | Show all your tracked pairs |
| `/settings` | Adjust alert thresholds |

---

## 📅 Roadmap

- [ ] Remove any batch summary reference
- [ ] Timeframe-specific alert types - limit short timeframes (1m, 5m) to only useful indicators like RSI and volume
- [ ] Interval-specific cooldowns - separate cooldown tracking per interval so alerts on different timeframes don't block each other
- [ ] Enhanced alert targeting - mention all users watching a particular symbol/interval instead of duplicating messages
- [ ] Make all command responses ephemeral 

- [ ] `/monitor` command to display permanent embeds with live crypto pair prices and current indicator statuses
- [ ] Config centralization - move all thresholds, periods, jitter %, etc. into a central YAML/JSON config for runtime tweaks
- [ ] Optional price change alerts
- [ ] Process separation - run the scheduler as a dedicated microservice to maintain alert timing during bot restarts

- [ ] Web dashboard (Flask/FastAPI)
- [ ] Telegram, email version/integrations
- [ ] Stocks integration (`yfinance`)

---

## 📜 License

MIT © [gabrielsaban](https://github.com/gabrielsaban)
