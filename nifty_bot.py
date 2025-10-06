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
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"

# Nifty 50 Config
NIFTY_50_SECURITY_ID = 13
NIFTY_SEGMENT = "IDX_I"

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
        self.current_expiry = None
        self.last_option_fetch = 0  # Rate limit tracker
        logger.info("Bot initialized successfully")
    
    def get_nearest_expiry(self):
        """Nearest expiry date मिळवतो"""
        try:
            payload = {
                "UnderlyingScrip": NIFTY_50_SECURITY_ID,
                "UnderlyingSeg": NIFTY_SEGMENT
            }
            
            response = requests.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'data' in data:
                    expiries = data['data']
                    if expiries:
                        self.current_expiry = expiries[0]  # Nearest expiry
                        logger.info(f"Nearest expiry: {self.current_expiry}")
                        return self.current_expiry
            
            return None
        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
            return None
    
    def get_nifty_ltp(self):
        """Nifty 50 चा LTP घेतो"""
        try:
            payload = {
                "IDX_I": [NIFTY_50_SECURITY_ID]
            }
            
            response = requests.post(
                DHAN_OHLC_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
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
                        
                        if result['close'] > 0:
                            result['change'] = ltp - result['close']
                            result['change_pct'] = (result['change'] / result['close']) * 100
                        else:
                            result['change'] = 0
                            result['change_pct'] = 0
                        
                        return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting LTP: {e}")
            return None
    
    def get_option_chain(self):
        """Option Chain data घेतो - ATM च्या जवळचे 10 strikes"""
        try:
            # Check expiry
            if not self.current_expiry:
                self.get_nearest_expiry()
            
            if not self.current_expiry:
                logger.warning("No expiry date found")
                return None
            
            payload = {
                "UnderlyingScrip": NIFTY_50_SECURITY_ID,
                "UnderlyingSeg": NIFTY_SEGMENT,
                "Expiry": self.current_expiry
            }
            
            response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            logger.info(f"Option Chain Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data:
                    spot_price = data['data'].get('last_price', 0)
                    oc_data = data['data'].get('oc', {})
                    
                    if not oc_data:
                        return None
                    
                    # Get all strikes and sort
                    strikes = sorted([float(s) for s in oc_data.keys()])
                    
                    # Find ATM strike (nearest to spot)
                    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
                    atm_index = strikes.index(atm_strike)
                    
                    # Get 5 strikes above and below ATM
                    start_idx = max(0, atm_index - 5)
                    end_idx = min(len(strikes), atm_index + 6)
                    selected_strikes = strikes[start_idx:end_idx]
                    
                    # Extract data for selected strikes
                    option_data = []
                    for strike in selected_strikes:
                        strike_key = f"{strike:.6f}"
                        strike_data = oc_data.get(strike_key, {})
                        
                        ce_data = strike_data.get('ce', {})
                        pe_data = strike_data.get('pe', {})
                        
                        option_data.append({
                            'strike': strike,
                            'ce_ltp': ce_data.get('last_price', 0),
                            'ce_oi': ce_data.get('oi', 0),
                            'pe_ltp': pe_data.get('last_price', 0),
                            'pe_oi': pe_data.get('oi', 0),
                            'is_atm': (strike == atm_strike)
                        })
                    
                    logger.info(f"Option Chain fetched: {len(option_data)} strikes")
                    return {
                        'spot': spot_price,
                        'atm': atm_strike,
                        'expiry': self.current_expiry,
                        'options': option_data
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    async def send_combined_message(self, nifty_data, option_data):
        """Nifty LTP + Option Chain combined message"""
        try:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            
            # Nifty Section
            change_emoji = "🟢" if nifty_data['change'] >= 0 else "🔴"
            change_sign = "+" if nifty_data['change'] >= 0 else ""
            
            message = f"📊 *NIFTY 50 LIVE*\n\n"
            message += f"💰 Spot: ₹{nifty_data['ltp']:,.2f}\n"
            
            if nifty_data['change'] != 0:
                message += f"{change_emoji} Change: {change_sign}{nifty_data['change']:,.2f} ({change_sign}{nifty_data['change_pct']:.2f}%)\n"
            
            # Option Chain Section
            if option_data:
                message += f"\n🎯 *OPTION CHAIN*\n"
                message += f"📅 Expiry: {option_data['expiry']}\n"
                message += f"🎲 ATM: ₹{option_data['atm']:,.0f}\n\n"
                
                message += "```\n"
                message += "Strike   CE-LTP  CE-OI    PE-LTP  PE-OI\n"
                message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                
                for opt in option_data['options']:
                    atm_mark = "🔸" if opt['is_atm'] else "  "
                    ce_ltp = f"{opt['ce_ltp']:6.1f}" if opt['ce_ltp'] > 0 else "  -   "
                    ce_oi = f"{opt['ce_oi']/1000:5.0f}K" if opt['ce_oi'] > 0 else "  -  "
                    pe_ltp = f"{opt['pe_ltp']:6.1f}" if opt['pe_ltp'] > 0 else "  -   "
                    pe_oi = f"{opt['pe_oi']/1000:5.0f}K" if opt['pe_oi'] > 0 else "  -  "
                    
                    message += f"{atm_mark}{opt['strike']:5.0f} {ce_ltp} {ce_oi}  {pe_ltp} {pe_oi}\n"
                
                message += "```"
            
            message += f"\n\n🕐 {timestamp}"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Combined message sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Bot started!")
        
        await self.send_startup_message()
        
        # Get expiry once at start
        self.get_nearest_expiry()
        
        iteration = 0
        
        while self.running:
            try:
                # Get Nifty LTP
                nifty_data = self.get_nifty_ltp()
                
                # Get Option Chain every 3 minutes (rate limit = 1 req/3 sec)
                option_data = None
                if iteration % 3 == 0:  # Every 3rd minute
                    option_data = self.get_option_chain()
                    await asyncio.sleep(3)  # Rate limit compliance
                
                if nifty_data:
                    await self.send_combined_message(nifty_data, option_data)
                else:
                    logger.warning("Could not fetch data")
                
                iteration += 1
                await asyncio.sleep(60)  # 1 minute interval
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = "🤖 *Nifty 50 + Option Chain Bot*\n\n"
            msg += "✅ Nifty LTP - Every minute\n"
            msg += "✅ Option Chain (10 strikes) - Every 3 minutes\n\n"
            msg += "🚂 Live on Railway.app"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


if __name__ == "__main__":
    try:
        if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
            logger.error("❌ Missing environment variables!")
            exit(1)
        
        bot = NiftyLTPBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
