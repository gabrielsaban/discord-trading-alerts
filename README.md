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

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/gabrielsaban/discord-trading-alerts.git
cd discord-trading-alerts
```

### 2. Set up virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```dotenv
DISCORD_BOT_TOKEN=your_discord_token
```

### 4. Run the bot

```bash
python main.py
```

---

## ⚙️ Commands

| Command | Description |
|--------|-------------|
| `/watch BTCUSDT` | Start tracking a pair |
| `/unwatch BTCUSDT` | Stop tracking |
| `/list` | Show all your tracked pairs |
| `/settings` | Adjust alert thresholds |

---

## 📦 Project Structure

```
discord-trading-alerts/
├── bot/
│   ├── indicators.py       # RSI, MACD, EMA logic
│   ├── binance.py          # API fetching & OHLC formatting
│   ├── alerts.py           # Trigger logic
│   ├── scheduler.py        # APScheduler integration
│   ├── db.py               # SQLite user storage
│   └── discord_bot.py      # Command logic
├── data/
│   └── alerts.db           # SQLite database
├── tests/
│   ├── test_indicators.py  # Unit tests for indicators
│   ├── test_binance.py     # Tests for the Binance API
│   └── conftest.py         # Test fixtures
├── .env
├── requirements.txt
├── README.md
└── main.py
```

---

## 📅 Roadmap

- [X] User-defined indicator thresholds
- [ ] `/monitor` command to display permanent embeds with live crypto pair prices and current indicator statuses
- [ ] `/status` command to show countdown timers until next check for each watched pair
- [X] Decoupled checking frequency - check all timeframes more frequently regardless of their interval
- [X] Global cooldown system to prevent duplicate alerts across timeframes
- [ ] Alert-based watching - users select specific alerts they want rather than intervals, and the system manages which timeframes to check
- [ ] Config centralization - move all thresholds, periods, jitter %, etc. into a central YAML/JSON config for runtime tweaks
- [ ] More customizable alert settings and notification options
- [ ] Process separation - run the scheduler as a dedicated microservice to maintain alert timing during bot restarts

- [ ] Web dashboard (Flask/FastAPI)
- [ ] Telegram version
- [ ] Stocks integration (`yfinance`)

---

## 📜 License

MIT © [gabrielsaban](https://github.com/gabrielsaban)