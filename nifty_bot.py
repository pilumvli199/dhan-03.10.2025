import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import json
from openai import OpenAI
from collections import deque

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration - Choose AI Provider
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "openai" or "groq"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize AI Client based on provider
if AI_PROVIDER == "groq":
    from groq import Groq
    ai_client = Groq(api_key=GROQ_API_KEY)
    AI_MODEL = "llama-3.1-70b-versatile"  # Free & fast
    logger.info("🤖 Using Groq (FREE)")
else:
    from openai import OpenAI
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    AI_MODEL = "gpt-4o-mini"
    logger.info("🤖 Using OpenAI")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ========================
# OPTIMIZED SETTINGS
# ========================
CANDLES_TO_FETCH = 200        # 16+ hours historical data
TECHNICAL_LOOKBACK = 50       # Support/Resistance calculation
ATR_PERIOD = 30               # ATR calculation period
AI_CANDLES_COUNT = 30         # Send to AI (last 2.5 hours)

# Stock/Index List
STOCKS_INDICES = {
    # Indices (Weekly Expiry Focus)
    "NIFTY 50": {"symbol": "Nifty 50", "segment": "IDX_I", "type": "index"},
    "NIFTY BANK": {"symbol": "Nifty Bank", "segment": "IDX_I", "type": "index"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I", "type": "index"},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "type": "index"},
    
    # High Volume Stocks (Monthly Expiry)
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ", "type": "stock"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ", "type": "stock"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ", "type": "stock"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ", "type": "stock"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ", "type": "stock"},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ", "type": "stock"},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ", "type": "stock"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ", "type": "stock"},
    "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ", "type": "stock"},
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ", "type": "stock"},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ", "type": "stock"},
    "LT": {"symbol": "LT", "segment": "NSE_EQ", "type": "stock"},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ", "type": "stock"},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ", "type": "stock"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ", "type": "stock"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ", "type": "stock"},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ", "type": "stock"},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ", "type": "stock"},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ", "type": "stock"},
    "TITAN": {"symbol": "TITAN", "segment": "NSE_EQ", "type": "stock"},
    "ASIANPAINT": {"symbol": "ASIANPAINT", "segment": "NSE_EQ", "type": "stock"},
}

# ========================
# AI OPTION BOT
# ========================

class AIOptionTradingBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_id_map = {}
        self.expiry_map = {}
        self.oi_history = {}
        self.last_option_chain_call = 0
        logger.info("🚀 OPTIMIZED AI Option Trading Bot initialized")
        logger.info(f"📊 Config: {CANDLES_TO_FETCH} candles | {TECHNICAL_LOOKBACK} lookback | {AI_CANDLES_COUNT} to AI")
    
    async def load_security_ids(self):
        """Load security IDs from Dhan"""
        try:
            logger.info("📥 Loading security IDs from Dhan...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
                
                for symbol, info in STOCKS_INDICES.items():
                    segment = info['segment']
                    symbol_name = info['symbol']
                    
                    for row in reader:
                        try:
                            # Index
                            if segment == "IDX_I":
                                if (row.get('SEM_SEGMENT') == 'I' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name,
                                            'type': info['type']
                                        }
                                        logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                        break
                            # Stock
                            else:
                                if (row.get('SEM_SEGMENT') == 'E' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name and
                                    row.get('SEM_EXM_EXCH_ID') == 'NSE'):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name,
                                            'type': info['type']
                                        }
                                        logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                        break
                        except Exception as e:
                            continue
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"📊 Total {len(self.security_id_map)} securities loaded")
                return True
            else:
                logger.error(f"❌ Failed to load instruments: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading security IDs: {e}")
            return False
    
    async def get_candle_data(self, security_id, segment):
        """Get 5-min candle data - OPTIMIZED"""
        try:
            # Fetch 10 days to ensure we get enough data
            from_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d 09:15:00")
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": segment,
                "instrument": "EQUITY" if segment == "NSE_EQ" else "INDEX",
                "interval": "5",
                "fromDate": from_date,
                "toDate": to_date
            }
            
            logger.info(f"📡 Fetching {CANDLES_TO_FETCH} candles (5-min)...")
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, dict) and 'open' in data and isinstance(data['open'], list):
                    candles = []
                    length = len(data['open'])
                    
                    for i in range(length):
                        candles.append({
                            'open': data['open'][i],
                            'high': data['high'][i],
                            'low': data['low'][i],
                            'close': data['close'][i],
                            'volume': data['volume'][i],
                            'timestamp': data.get('timestamp', [0]*length)[i]
                        })
                    
                    # Return last CANDLES_TO_FETCH candles (200 by default)
                    logger.info(f"✅ Got {len(candles)} total candles, using last {CANDLES_TO_FETCH}")
                    return candles[-CANDLES_TO_FETCH:]
                
                logger.warning(f"⚠️ Unexpected response format")
            else:
                logger.error(f"❌ Candle API failed: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting candle data: {e}")
            return None
    
    def get_all_expiries(self, security_id, segment):
        """Get ALL expiries for a symbol with retry logic"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {
                    "UnderlyingScrip": security_id,
                    "UnderlyingSeg": segment
                }
                
                logger.info(f"🔍 Fetching expiries (attempt {attempt+1}/{max_retries})")
                
                response = requests.post(
                    DHAN_EXPIRY_LIST_URL,
                    json=payload,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == 'success' and data.get('data'):
                        expiries = data['data']
                        logger.info(f"✅ Found {len(expiries)} expiries: {expiries[:5]}")
                        return expiries
                    else:
                        logger.warning(f"⚠️ Expiry response issue: {data}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)
                            continue
                elif response.status_code == 429:
                    logger.warning(f"⚠️ Rate limit hit, waiting 5s...")
                    import time
                    time.sleep(5)
                    if attempt < max_retries - 1:
                        continue
                else:
                    logger.error(f"❌ Expiry API failed: {response.status_code}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)
                        continue
                
            except Exception as e:
                logger.error(f"❌ Error getting expiry: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
        
        return []
    
    def select_best_expiry(self, symbol, expiry_list, symbol_type):
        """Intelligent expiry selection"""
        try:
            if not expiry_list:
                return None
            
            today = datetime.now().date()
            
            future_expiries = []
            for e in expiry_list:
                try:
                    expiry_date = datetime.strptime(e, '%Y-%m-%d').date()
                    if expiry_date >= today:
                        future_expiries.append(expiry_date)
                except:
                    continue
            
            if not future_expiries:
                logger.warning(f"⚠️ {symbol}: No future expiries")
                return None
            
            future_expiries.sort()
            
            if symbol_type == 'index':
                selected = future_expiries[0]
                days_to_expiry = (selected - today).days
                logger.info(f"📅 {symbol}: Weekly expiry = {selected} ({days_to_expiry} days)")
            else:
                monthly_expiries = [e for e in future_expiries if e.day >= 20]
                
                if monthly_expiries:
                    selected = monthly_expiries[0]
                    days_to_expiry = (selected - today).days
                    logger.info(f"📅 {symbol}: Monthly expiry = {selected} ({days_to_expiry} days)")
                else:
                    selected = future_expiries[0]
                    days_to_expiry = (selected - today).days
                    logger.warning(f"⚠️ {symbol}: Using nearest = {selected} ({days_to_expiry} days)")
            
            return selected.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"❌ Error selecting expiry: {e}")
            return None
    
    def update_expiry_for_symbol(self, symbol, security_id, segment, symbol_type):
        """Update expiry with auto-rollover"""
        try:
            expiry_list = self.get_all_expiries(security_id, segment)
            
            if not expiry_list:
                logger.warning(f"⚠️ {symbol}: No expiries from API")
                return None
            
            selected_expiry = self.select_best_expiry(symbol, expiry_list, symbol_type)
            
            if not selected_expiry:
                return None
            
            if symbol in self.expiry_map:
                old_expiry = self.expiry_map[symbol]
                if old_expiry != selected_expiry:
                    logger.warning(f"🔄 {symbol}: Rollover {old_expiry} → {selected_expiry}")
            
            self.expiry_map[symbol] = selected_expiry
            return selected_expiry
            
        except Exception as e:
            logger.error(f"❌ Error updating expiry: {e}")
            return None
    
    async def get_option_chain_safe(self, security_id, segment, expiry):
        """Rate-limit safe option chain fetch"""
        try:
            import time
            current_time = time.time()
            time_since_last = current_time - self.last_option_chain_call
            
            if time_since_last < 3:
                sleep_time = 3 - time_since_last
                logger.info(f"⏳ Rate limit: Waiting {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
            
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            self.last_option_chain_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    logger.info(f"✅ Option chain received")
                    return data['data']
                else:
                    logger.warning(f"⚠️ No data in response")
            else:
                logger.error(f"❌ Option Chain failed: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting option chain: {e}")
            return None
    
    def calculate_technical_indicators(self, candles):
        """Calculate technical indicators - OPTIMIZED with more data"""
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            # Use TECHNICAL_LOOKBACK (50 candles = 4+ hours)
            lookback_highs = highs[-TECHNICAL_LOOKBACK:]
            lookback_lows = lows[-TECHNICAL_LOOKBACK:]
            
            resistance = max(lookback_highs)
            support = min(lookback_lows)
            
            # ATR with ATR_PERIOD (30 candles = 2.5 hours)
            tr_list = []
            atr_period = min(ATR_PERIOD, len(candles) - 1)
            
            for i in range(1, atr_period + 1):
                high_low = highs[-i] - lows[-i]
                high_close = abs(highs[-i] - closes[-i-1])
                low_close = abs(lows[-i] - closes[-i-1])
                tr = max(high_low, high_close, low_close)
                tr_list.append(tr)
            
            atr = sum(tr_list) / len(tr_list) if tr_list else 0
            
            # Price change over full data
            price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            # Volume analysis (last 50 candles)
            avg_volume = sum(volumes[-50:]) / 50
            current_volume = volumes[-1]
            volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            # Additional: Trend strength
            short_ma = sum(closes[-10:]) / 10  # 50-min MA
            long_ma = sum(closes[-30:]) / 30   # 150-min MA
            trend = "BULLISH" if short_ma > long_ma else "BEARISH"
            
            # Volatility (std dev of last 30 closes)
            mean_price = sum(closes[-30:]) / 30
            variance = sum([(x - mean_price)**2 for x in closes[-30:]]) / 30
            volatility = variance ** 0.5
            
            return {
                "current_price": closes[-1],
                "support": support,
                "resistance": resistance,
                "atr": atr,
                "price_change_pct": price_change_pct,
                "volume_spike": volume_spike,
                "avg_volume": avg_volume,
                "trend": trend,
                "volatility": volatility,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "data_points": len(candles)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return None
    
    def analyze_option_chain_advanced(self, oc_data, spot_price, symbol):
        """Advanced option chain analysis"""
        try:
            oc = oc_data.get('oc', {})
            if not oc:
                return None
            
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_volume = 0
            total_pe_volume = 0
            
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            relevant_strikes = strikes[start_idx:end_idx]
            
            max_ce_oi_strike = None
            max_pe_oi_strike = None
            max_ce_oi = 0
            max_pe_oi = 0
            
            strike_wise_data = {}
            
            for strike in relevant_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                ce_oi = ce.get('oi', 0)
                pe_oi = pe.get('oi', 0)
                ce_vol = ce.get('volume', 0)
                pe_vol = pe.get('volume', 0)
                
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
                total_ce_volume += ce_vol
                total_pe_volume += pe_vol
                
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    max_ce_oi_strike = strike
                
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    max_pe_oi_strike = strike
                
                strike_wise_data[strike] = {
                    'ce_oi': ce_oi,
                    'pe_oi': pe_oi,
                    'ce_vol': ce_vol,
                    'pe_vol': pe_vol
                }
            
            timestamp = datetime.now().isoformat()
            
            if symbol not in self.oi_history:
                self.oi_history[symbol] = deque(maxlen=5)
            
            self.oi_history[symbol].append({
                'timestamp': timestamp,
                'ce_oi': total_ce_oi,
                'pe_oi': total_pe_oi,
                'strikes': strike_wise_data
            })
            
            oi_change_pct = 0
            if len(self.oi_history[symbol]) >= 2:
                old_ce_oi = self.oi_history[symbol][0]['ce_oi']
                old_pe_oi = self.oi_history[symbol][0]['pe_oi']
                
                ce_change = ((total_ce_oi - old_ce_oi) / old_ce_oi * 100) if old_ce_oi > 0 else 0
                pe_change = ((total_pe_oi - old_pe_oi) / old_pe_oi * 100) if old_pe_oi > 0 else 0
                
                oi_change_pct = (ce_change + pe_change) / 2
            
            pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0
            
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            atm_ce = atm_data.get('ce', {})
            atm_pe = atm_data.get('pe', {})
            
            return {
                "pcr": pcr,
                "atm_strike": atm_strike,
                "max_ce_oi_strike": max_ce_oi_strike,
                "max_pe_oi_strike": max_pe_oi_strike,
                "ce_total_oi": total_ce_oi,
                "pe_total_oi": total_pe_oi,
                "ce_total_volume": total_ce_volume,
                "pe_total_volume": total_pe_volume,
                "atm_ce_price": atm_ce.get('last_price', 0),
                "atm_pe_price": atm_pe.get('last_price', 0),
                "atm_ce_iv": atm_ce.get('implied_volatility', 0),
                "atm_pe_iv": atm_pe.get('implied_volatility', 0),
                "oi_change_pct": oi_change_pct,
                "oi_snapshots": len(self.oi_history[symbol])
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing option chain: {e}")
            return None
    
    async def get_ai_analysis(self, symbol, candles, technical_data, option_data, spot_price):
        """GPT analysis - FIXED TO USE OPTION PREMIUMS"""
        try:
            # Send last AI_CANDLES_COUNT (30 by default = 2.5 hours)
            recent_candles = candles[-AI_CANDLES_COUNT:]
            
            # Simplified candle data to save tokens
            candles_summary = [
                {
                    "o": round(c.get('open'), 2),
                    "h": round(c.get('high'), 2),
                    "l": round(c.get('low'), 2),
                    "c": round(c.get('close'), 2),
                    "v": int(c.get('volume', 0))
                }
                for c in recent_candles
            ]
            
            # Get actual option premiums
            atm_ce_premium = option_data.get('atm_ce_price', 0)
            atm_pe_premium = option_data.get('atm_pe_price', 0)
            
            prompt = f"""You are an expert option trader. Analyze {symbol} and provide trading signal.

**CRITICAL RULES:**
1. Use OPTION PREMIUM prices (NOT stock/index prices)
2. Entry, Stop Loss, Target must be OPTION PREMIUMS
3. Calculate realistic R:R based on option premiums
4. Use ATM option prices as reference

**Market Data:**

Underlying Spot Price: ₹{spot_price:,.2f}

**Technical Analysis ({technical_data.get('data_points', 0)} candles analyzed):**
- Trend: {technical_data['trend']}
- Support: ₹{technical_data['support']:,.2f}
- R:R calculation: ({atm_ce_premium * 1.8:.2f} - {atm_ce_premium:.2f}) / ({atm_ce_premium:.2f} - {max(atm_ce_premium * 0.65, 0.5):.2f}) = Risk:Reward ratio

**Respond ONLY with this JSON format:**

{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 70-95,
    "strike": {option_data['atm_strike']},
    "entry_price": {atm_ce_premium:.2f},
    "stop_loss": {max(atm_ce_premium * 0.65, 0.5):.2f},
    "target": {atm_ce_premium * 1.8:.2f},
    "risk_reward": 2.5,
    "reasoning": "Combine technical trend + PCR + OI data in 1-2 lines"
}}

**STRICT VALIDATION:**
- entry_price MUST be the current ATM option premium (₹{atm_ce_premium:.2f} for CE or ₹{atm_pe_premium:.2f} for PE)
- stop_loss MUST be lower than entry_price (typically 30-40% down)
- target MUST be higher than entry_price (typically 60-100% up)
- risk_reward MUST be calculated as: (target - entry) / (entry - stop_loss)
- risk_reward MUST be >= 2.0
- confidence MUST be >= 70
- DO NOT use stock/index prices like ₹{spot_price:,.2f}

Respond with JSON ONLY, no other text.
"""

            response = ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert options trader. Always respond with valid JSON only. Use OPTION PREMIUMS, never stock prices for entry/SL/target."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            # Parse AI response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON
            analysis = json.loads(content)
            
            # Validate the response
            required_fields = ["signal", "confidence", "strike", "entry_price", "stop_loss", "target", "risk_reward", "reasoning"]
            if not all(field in analysis for field in required_fields):
                logger.error(f"❌ AI response missing required fields")
                return None
            
            # Validate option premium logic
            if analysis['signal'] in ['BUY_CE', 'BUY_PE']:
                if analysis['stop_loss'] >= analysis['entry_price']:
                    logger.error(f"❌ Invalid: Stop loss must be < entry price")
                    return None
                if analysis['target'] <= analysis['entry_price']:
                    logger.error(f"❌ Invalid: Target must be > entry price")
                    return None
                if analysis['confidence'] < 70:
                    logger.error(f"❌ Invalid: Confidence too low")
                    return None
            
            logger.info(f"✅ AI Analysis: {analysis['signal']} @ ₹{analysis.get('entry_price', 0):.2f}")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return None
    
    async def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
            logger.info("✅ Message sent to Telegram")
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
    
    async def analyze_symbol(self, symbol):
        """Complete analysis for one symbol"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 ANALYZING: {symbol}")
            logger.info(f"{'='*60}")
            
            if symbol not in self.security_id_map:
                logger.error(f"❌ {symbol}: Security ID not found")
                return
            
            sec_info = self.security_id_map[symbol]
            security_id = sec_info['security_id']
            segment = sec_info['segment']
            symbol_type = sec_info['type']
            
            # Step 1: Get candle data
            candles = await self.get_candle_data(security_id, segment)
            if not candles or len(candles) < 50:
                logger.error(f"❌ {symbol}: Insufficient candle data")
                return
            
            # Step 2: Calculate technical indicators
            technical_data = self.calculate_technical_indicators(candles)
            if not technical_data:
                logger.error(f"❌ {symbol}: Failed to calculate indicators")
                return
            
            spot_price = technical_data['current_price']
            logger.info(f"💹 {symbol} Spot: ₹{spot_price:,.2f} | Trend: {technical_data['trend']}")
            
            # Step 3: Update expiry
            expiry = self.update_expiry_for_symbol(symbol, security_id, segment, symbol_type)
            if not expiry:
                logger.error(f"❌ {symbol}: Failed to get expiry")
                return
            
            # Step 4: Get option chain
            oc_data = await self.get_option_chain_safe(security_id, segment, expiry)
            if not oc_data:
                logger.error(f"❌ {symbol}: Failed to get option chain")
                return
            
            # Step 5: Analyze option chain
            option_data = self.analyze_option_chain_advanced(oc_data, spot_price, symbol)
            if not option_data:
                logger.error(f"❌ {symbol}: Failed to analyze option chain")
                return
            
            logger.info(f"📊 PCR: {option_data['pcr']:.2f} | ATM: ₹{option_data['atm_strike']:,.0f}")
            logger.info(f"💰 CE Premium: ₹{option_data['atm_ce_price']:.2f} | PE Premium: ₹{option_data['atm_pe_price']:.2f}")
            
            # Step 6: Get AI analysis
            ai_analysis = await self.get_ai_analysis(symbol, candles, technical_data, option_data, spot_price)
            
            if not ai_analysis:
                logger.warning(f"⚠️ {symbol}: No AI signal generated")
                return
            
            # Step 7: Send signal if BUY
            if ai_analysis['signal'] in ['BUY_CE', 'BUY_PE']:
                await self.send_trade_signal(symbol, ai_analysis, technical_data, option_data, expiry)
            else:
                logger.info(f"⏸️ {symbol}: NO TRADE (Confidence: {ai_analysis.get('confidence', 0)}%)")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
    
    async def send_trade_signal(self, symbol, ai_analysis, technical_data, option_data, expiry):
        """Format and send trading signal"""
        try:
            signal_type = "📈 CALL" if ai_analysis['signal'] == 'BUY_CE' else "📉 PUT"
            
            message = f"""
🎯 <b>{signal_type} SIGNAL - {symbol}</b>

<b>📊 Market Info:</b>
• Spot Price: ₹{technical_data['current_price']:,.2f}
• Trend: {technical_data['trend']}
• PCR: {option_data['pcr']:.2f}

<b>🎫 Option Details:</b>
• Strike: ₹{ai_analysis['strike']:,.0f}
• Expiry: {expiry}
• Entry: ₹{ai_analysis['entry_price']:.2f}
• Stop Loss: ₹{ai_analysis['stop_loss']:.2f}
• Target: ₹{ai_analysis['target']:.2f}
• R:R Ratio: {ai_analysis['risk_reward']:.2f}

<b>🤖 AI Confidence: {ai_analysis['confidence']}%</b>

<b>💡 Reasoning:</b>
{ai_analysis['reasoning']}

<b>⚠️ Risk Management:</b>
• Max Loss: ₹{ai_analysis['entry_price'] - ai_analysis['stop_loss']:.2f} per lot
• Expected Gain: ₹{ai_analysis['target'] - ai_analysis['entry_price']:.2f} per lot

<i>⏰ {datetime.now().strftime('%d-%m-%Y %I:%M %p')}</i>
"""
            
            await self.send_telegram_message(message)
            logger.info(f"✅ {symbol}: Trade signal sent!")
            
        except Exception as e:
            logger.error(f"❌ Error sending trade signal: {e}")
    
    async def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        try:
            logger.info(f"\n{'#'*80}")
            logger.info(f"🚀 STARTING ANALYSIS CYCLE - {datetime.now().strftime('%I:%M %p')}")
            logger.info(f"{'#'*80}\n")
            
            # Analyze all symbols
            for symbol in STOCKS_INDICES.keys():
                await self.analyze_symbol(symbol)
                await asyncio.sleep(2)  # Pause between symbols
            
            logger.info(f"\n{'#'*80}")
            logger.info(f"✅ CYCLE COMPLETE - {datetime.now().strftime('%I:%M %p')}")
            logger.info(f"{'#'*80}\n")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {e}")
    
    async def start(self):
        """Main bot loop"""
        try:
            # Load security IDs once
            success = await self.load_security_ids()
            if not success:
                logger.error("❌ Failed to load security IDs. Exiting.")
                return
            
            # Send startup message
            await self.send_telegram_message("🚀 <b>AI Option Trading Bot Started!</b>\n\nAnalyzing markets...")
            
            # Run analysis every 15 minutes during market hours
            while self.running:
                current_time = datetime.now()
                
                # Check if market hours (9:15 AM - 3:30 PM)
                if current_time.weekday() < 5:  # Monday to Friday
                    market_start = current_time.replace(hour=9, minute=15, second=0)
                    market_end = current_time.replace(hour=15, minute=30, second=0)
                    
                    if market_start <= current_time <= market_end:
                        await self.run_analysis_cycle()
                        
                        # Wait 15 minutes
                        logger.info("⏳ Waiting 15 minutes for next cycle...")
                        await asyncio.sleep(900)  # 15 minutes
                    else:
                        logger.info("⏸️ Outside market hours. Waiting...")
                        await asyncio.sleep(300)  # 5 minutes
                else:
                    logger.info("⏸️ Weekend. Waiting...")
                    await asyncio.sleep(3600)  # 1 hour
                    
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
        finally:
            await self.send_telegram_message("🛑 <b>Bot Stopped</b>")

# ========================
# MAIN
# ========================

async def main():
    bot = AIOptionTradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())esistance: ₹{technical_data['resistance']:,.2f}
- ATR: ₹{technical_data['atr']:.2f}
- Price Change: {technical_data['price_change_pct']:.2f}%
- Volume Spike: {technical_data['volume_spike']:.2f}x
- Volatility: ₹{technical_data['volatility']:.2f}
- Short MA (50min): ₹{technical_data['short_ma']:.2f}
- Long MA (150min): ₹{technical_data['long_ma']:.2f}

**Recent {AI_CANDLES_COUNT} Candles (Last 2.5 hours):**
{json.dumps(candles_summary[:10], indent=2)}
...and {len(candles_summary)-10} more candles

**Options Chain Data:**
- ATM Strike: ₹{option_data['atm_strike']:,.0f}
- ATM CALL Premium: ₹{atm_ce_premium:.2f}
- ATM PUT Premium: ₹{atm_pe_premium:.2f}
- ATM CE IV: {option_data.get('atm_ce_iv', 0):.1f}%
- ATM PE IV: {option_data.get('atm_pe_iv', 0):.1f}%
- PCR Ratio: {option_data['pcr']:.2f}
- Max CE OI Strike: ₹{option_data.get('max_ce_oi_strike', 0):,.0f}
- Max PE OI Strike: ₹{option_data.get('max_pe_oi_strike', 0):,.0f}
- Total CE OI: {option_data['ce_total_oi']:,}
- Total PE OI: {option_data['pe_total_oi']:,}
- CE Volume: {option_data['ce_total_volume']:,}
- PE Volume: {option_data['pe_total_volume']:,}
- OI Change: {option_data.get('oi_change_pct', 0):.2f}%

**EXAMPLE (MUST FOLLOW THIS FORMAT):**

If recommending BUY_CE with ATM strike ₹{option_data['atm_strike']:,.0f}:
- Current ATM CE Premium: ₹{atm_ce_premium:.2f}
- Entry: ₹{atm_ce_premium:.2f} (current premium)
- Stop Loss: ₹{max(atm_ce_premium * 0.65, 0.5):.2f} (35% loss max)
- Target: ₹{atm_ce_premium * 1.8:.2f} (80% gain)
- R
