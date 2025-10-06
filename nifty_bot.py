import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
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
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"

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
                        self.current_expiry = expiries[0]
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
    
    def get_historical_data(self, days=5):
        """Historical data घेतो (last N days)"""
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            payload = {
                "securityId": str(NIFTY_50_SECURITY_ID),
                "exchangeSegment": NIFTY_SEGMENT,
                "instrument": "INDEX",
                "expiryCode": 0,
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                DHAN_HISTORICAL_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    logger.info(f"Historical data fetched: {len(data['data'])} days")
                    return data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def get_option_chain_with_greeks(self):
        """Option Chain data with Greeks घेतो"""
        try:
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
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success' and 'data' in data:
                    spot_price = data['data'].get('last_price', 0)
                    oc_data = data['data'].get('oc', {})
                    
                    if not oc_data:
                        return None
                    
                    # Get all strikes and sort
                    strikes = sorted([float(s) for s in oc_data.keys()])
                    
                    # Find ATM
                    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
                    atm_index = strikes.index(atm_strike)
                    
                    # Get 5 strikes around ATM
                    start_idx = max(0, atm_index - 5)
                    end_idx = min(len(strikes), atm_index + 6)
                    selected_strikes = strikes[start_idx:end_idx]
                    
                    # Extract full data with Greeks
                    option_data = []
                    for strike in selected_strikes:
                        strike_key = f"{strike:.6f}"
                        strike_data = oc_data.get(strike_key, {})
                        
                        ce_data = strike_data.get('ce', {})
                        pe_data = strike_data.get('pe', {})
                        
                        # Extract Greeks
                        ce_greeks = ce_data.get('greeks', {})
                        pe_greeks = pe_data.get('greeks', {})
                        
                        option_data.append({
                            'strike': strike,
                            'ce_ltp': ce_data.get('last_price', 0),
                            'ce_oi': ce_data.get('oi', 0),
                            'ce_volume': ce_data.get('volume', 0),
                            'ce_iv': ce_data.get('implied_volatility', 0),
                            'ce_delta': ce_greeks.get('delta', 0),
                            'ce_theta': ce_greeks.get('theta', 0),
                            'ce_gamma': ce_greeks.get('gamma', 0),
                            'ce_vega': ce_greeks.get('vega', 0),
                            'pe_ltp': pe_data.get('last_price', 0),
                            'pe_oi': pe_data.get('oi', 0),
                            'pe_volume': pe_data.get('volume', 0),
                            'pe_iv': pe_data.get('implied_volatility', 0),
                            'pe_delta': pe_greeks.get('delta', 0),
                            'pe_theta': pe_greeks.get('theta', 0),
                            'pe_gamma': pe_greeks.get('gamma', 0),
                            'pe_vega': pe_greeks.get('vega', 0),
                            'is_atm': (strike == atm_strike)
                        })
                    
                    logger.info(f"Option Chain with Greeks fetched: {len(option_data)} strikes")
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
    
    async def send_option_chain_message(self, option_data):
        """Option Chain with basic data"""
        try:
            message = f"📊 *OPTION CHAIN*\n"
            message += f"📅 Expiry: {option_data['expiry']}\n"
            message += f"💰 Spot: ₹{option_data['spot']:,.2f}\n"
            message += f"🎯 ATM: ₹{option_data['atm']:,.0f}\n\n"
            
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
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Option chain message sent")
            
        except Exception as e:
            logger.error(f"Error sending option chain: {e}")
    
    async def send_greeks_message(self, option_data):
        """Greeks data (Delta, Theta, Gamma)"""
        try:
            # Find ATM option
            atm_option = next((opt for opt in option_data['options'] if opt['is_atm']), None)
            
            if not atm_option:
                return
            
            message = f"🎲 *GREEKS - ATM Strike {atm_option['strike']:.0f}*\n\n"
            
            message += "```\n"
            message += "         CALL              PUT\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"Delta:  {atm_option['ce_delta']:7.4f}      {atm_option['pe_delta']:7.4f}\n"
            message += f"Theta:  {atm_option['ce_theta']:7.2f}      {atm_option['pe_theta']:7.2f}\n"
            message += f"Gamma:  {atm_option['ce_gamma']:7.5f}      {atm_option['pe_gamma']:7.5f}\n"
            message += f"Vega:   {atm_option['ce_vega']:7.2f}      {atm_option['pe_vega']:7.2f}\n"
            message += f"IV:     {atm_option['ce_iv']:7.2f}%     {atm_option['pe_iv']:7.2f}%\n"
            message += "```\n\n"
            
            message += "📝 *Greek Meanings:*\n"
            message += "• Delta: Price change per ₹1 move\n"
            message += "• Theta: Time decay per day\n"
            message += "• Gamma: Delta change rate\n"
            message += "• Vega: IV sensitivity\n"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Greeks message sent")
            
        except Exception as e:
            logger.error(f"Error sending Greeks: {e}")
    
    async def send_historical_message(self, historical_data):
        """Historical data message (last 5 days)"""
        try:
            message = f"📈 *HISTORICAL DATA (Last 5 Days)*\n\n"
            
            message += "```\n"
            message += "Date       Open    High     Low   Close\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            for day in historical_data[-5:]:
                date = day.get('timestamp', '')[:10]
                open_p = day.get('open', 0)
                high = day.get('high', 0)
                low = day.get('low', 0)
                close = day.get('close', 0)
                
                message += f"{date} {open_p:7.2f} {high:7.2f} {low:6.2f} {close:7.2f}\n"
            
            message += "```"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Historical data sent")
            
        except Exception as e:
            logger.error(f"Error sending historical: {e}")
    
    async def send_nifty_ltp_message(self, data):
        """Quick Nifty LTP update"""
        try:
            change_emoji = "🟢" if data['change'] >= 0 else "🔴"
            change_sign = "+" if data['change'] >= 0 else ""
            
            message = f"📊 *NIFTY 50*\n"
            message += f"💰 {data['ltp']:,.2f} "
            message += f"{change_emoji} {change_sign}{data['change']:,.2f} ({change_sign}{data['change_pct']:.2f}%)"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error sending LTP: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Bot started!")
        
        await self.send_startup_message()
        
        # Get expiry
        self.get_nearest_expiry()
        
        # Send historical data once at start
        hist_data = self.get_historical_data(5)
        if hist_data:
            await self.send_historical_message(hist_data)
        
        iteration = 0
        
        while self.running:
            try:
                # Nifty LTP - every minute
                nifty_data = self.get_nifty_ltp()
                if nifty_data:
                    await self.send_nifty_ltp_message(nifty_data)
                
                # Option Chain - every 3 minutes
                if iteration % 3 == 0:
                    option_data = self.get_option_chain_with_greeks()
                    if option_data:
                        await self.send_option_chain_message(option_data)
                        await asyncio.sleep(2)  # Small delay
                        await self.send_greeks_message(option_data)
                    
                    await asyncio.sleep(3)  # API rate limit
                
                iteration += 1
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup"""
        try:
            msg = "🤖 *Nifty Pro Bot Started!*\n\n"
            msg += "✅ Live LTP - Every minute\n"
            msg += "✅ Option Chain - Every 3 min\n"
            msg += "✅ Greeks (Delta/Theta/Gamma)\n"
            msg += "✅ Historical Data (5 days)\n\n"
            msg += "🚂 Railway.app"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Startup error: {e}")


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
