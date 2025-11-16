import asyncio
import requests
import pandas as pd
import numpy as np
import openai
import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ----------------------------
# Your Keys (Directly Added)
# ----------------------------
BOT_TOKEN = "8463339584:AAFrCyAUirjlN4EMlcL236x0_FyLl4InXJ0"
OPENAI_API_KEY = "sk-proj-MUEjFM9U6UqqVZtzxMm5cWei7WYK1f2A-wDJfPtr_5Ui5ux81kykgaXDWuOOCUUKOceIxknOt3BlbkFJSMH6u1XmMz-49gm_jvP8-mPfEsa3VPFJwYkJH_w4i41BJd5ihaYUjpxvthLQTBD3wiSwq_gKgA"
ALPHAVANTAGE_KEY = "CT9DC5EEEEYES4YP"
TELEGRAM_CHAT_ID = 674022022  # Replace with your Telegram chat ID

openai.api_key = OPENAI_API_KEY

# ----------------------------
# Fetch crypto price data
# ----------------------------
def get_price(symbol, interval="5min"):
    url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=USD&interval={interval}&apikey={ALPHAVANTAGE_KEY}"
    r = requests.get(url).json()
    try:
        df = pd.DataFrame(r[f"Time Series Crypto ({interval})"]).T
        df = df.astype(float)
        return df
    except:
        return None

# ----------------------------
# Compute RSI
# ----------------------------
def compute_rsi(df, length=14):
    close = df["4. close"]
    delta = close.diff()
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# ----------------------------
# Generate AI signal
# ----------------------------
def generate_signal(symbol, rsi):
    prompt = f"""
    You are a crypto trading bot. Use these:
    - RSI = {rsi}
    - Asset: {symbol}

    Generate:
    • Buy or Sell signal
    • Entry Price
    • Take Profit (TP)
    • Stop Loss (SL)
    • Short reason
    """
    ai = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return ai.choices[0].message.content

# ----------------------------
# Send message to Telegram
# ----------------------------
def send_message(text):
    bot = telegram.Bot(token=BOT_TOKEN)
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

# ----------------------------
# Auto-signal function
# ----------------------------
async def auto_signal():
    await asyncio.sleep(10)
    symbols = ["BTC", "ETH", "SOL"]
    intervals = ["5min", "15min", "60min", "240min"]  # multi-timeframes

    while True:
        for symbol in symbols:
            for interval in intervals:
                df = get_price(symbol, interval)
                if df is None:
                    continue
                rsi = compute_rsi(df)
                result = generate_signal(symbol, rsi)
                send_message(f"{symbol} {interval} Signal:\n{result}")
        await asyncio.sleep(3600)  # run every hour

# ----------------------------
# Start command
# ----------------------------
async def start(update: Update, context):
    await update.message.reply_text(
        "🤖 AI Crypto Signal Bot Ready!\nAuto-signals running every hour for BTC, ETH, SOL."
    )

# ----------------------------
# Main app
# ----------------------------
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))

# Run bot with auto-signal
async def main():
    await asyncio.gather(
        app.run_polling(),
        auto_signal()
    )

asyncio.run(main())
