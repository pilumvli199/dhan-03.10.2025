import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime
import logging

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# CONFIGURATION
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_LTP_URL = f"{DHAN_API_BASE}/v2/marketfeed/ltp"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"

# Nifty 50 Index Security ID
NIFTY_50_SECURITY_ID = 13

# ========================
# BOT CODE
# ========================

class NiftyLTPBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        logger.info("Bot initialized successfully")
    
    def get_nifty_ltp(self):
        """Dhan REST API वरून Nifty 50 चा LTP घेतो"""
        try:
            # Request body for Nifty 50 Index
            payload = {
                "IDX_I": [NIFTY_50_SECURITY_ID]
            }
            
            # Get OHLC data (includes LTP)
            response = requests.post(
                DHAN_OHLC_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            logger.info(f"API Status Code: {response.status_code}")
            logger.info(f"API Response: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data:
                    idx_data = data['data'].get('IDX_I', {})
                    nifty_data = idx_data.get(str(NIFTY_50_SECURITY_ID), {})
                    
                    if nifty_data and 'last_price' in nifty_data:
                        ltp = nifty_data['last_price']
                        ohlc = nifty_data.get('ohlc', {})
                        
                        result = {
                            'ltp': ltp,
                            'open': ohlc.get('open', 0),
                            'high': ohlc.get('high', 0),
                            'low': ohlc.get('low', 0),
                            'close': ohlc.get('close', 0)
                        }
                        
                        # Calculate change
                        if result['close'] > 0:
                            result['change'] = ltp - result['close']
                            result['change_pct'] = (result['change'] / result['close']) * 100
                        else:
                            result['change'] = 0
                            result['change_pct'] = 0
                        
                        logger.info(f"LTP fetched successfully: {ltp}")
                        return result
            
            logger.warning(f"API returned non-success response: {response.status_code}")
            return None
            
        except requests.exceptions.Timeout:
            logger.error("API request timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting LTP: {e}")
            return None
    
    async def send_ltp_message(self, data):
        """Telegram वर LTP पाठवतो"""
        try:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            
            # Change indicator
            change_emoji = "🟢" if data['change'] >= 0 else "🔴"
            change_sign = "+" if data['change'] >= 0 else ""
            
            message = f"📊 *NIFTY 50 LIVE*\n\n"
            message += f"💰 LTP: ₹{data['ltp']:,.2f}\n"
            
            if data['change'] != 0:
                message += f"{change_emoji} Change: {change_sign}{data['change']:,.2f} ({change_sign}{data['change_pct']:.2f}%)\n\n"
            
            if data['open'] > 0:
                message += f"🔵 Open: ₹{data['open']:,.2f}\n"
            if data['high'] > 0:
                message += f"📈 High: ₹{data['high']:,.2f}\n"
            if data['low'] > 0:
                message += f"📉 Low: ₹{data['low']:,.2f}\n"
            if data['close'] > 0:
                message += f"⚪ Prev Close: ₹{data['close']:,.2f}\n"
            
            message += f"\n🕐 Time: {timestamp}\n"
            message += f"_Updated every minute_ ⏱️"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"Message sent - LTP: {data['ltp']}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def run(self):
        """Main loop - दर मिनिटाला LTP पाठवतो"""
        logger.info("🚀 Bot started! Sending Nifty 50 LTP every minute...")
        
        await self.send_startup_message()
        
        while self.running:
            try:
                data = self.get_nifty_ltp()
                
                if data:
                    await self.send_ltp_message(data)
                else:
                    logger.warning("Could not fetch LTP - Market might be closed or API issue")
                
                # 1 minute wait
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Bot सुरू झाल्यावर message पाठवतो"""
        try:
            msg = "🤖 *Nifty 50 LTP Bot Started!*\n\n"
            msg += "तुम्हाला आता दर मिनिटाला Nifty 50 चा Live LTP मिळेल! 📈\n\n"
            msg += "✅ Powered by Dhan API v2 (REST)\n"
            msg += "🚂 Deployed on Railway.app\n\n"
            msg += "_Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)_"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


# ========================
# BOT RUN करा
# ========================
if __name__ == "__main__":
    try:
        # Environment variables check
        if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
            logger.error("❌ Missing environment variables!")
            logger.error("Please set: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN")
            exit(1)
        
        bot = NiftyLTPBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
