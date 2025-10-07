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

# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

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
        logger.info("🚀 Advanced AI Option Trading Bot initialized")
    
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
    
    async def get_multi_timeframe_data(self, security_id, segment):
        """Get 5-min candle data"""
        try:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d 09:15:00")
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": segment,
                "instrument": "EQUITY" if segment == "NSE_EQ" else "INDEX",
                "interval": "5",
                "fromDate": from_date,
                "toDate": to_date
            }
            
            logger.info(f"📡 Fetching 5min data...")
            
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
                    logger.info(f"✅ Got {length} candles")
                    
                    for i in range(length):
                        candles.append({
                            'open': data['open'][i],
                            'high': data['high'][i],
                            'low': data['low'][i],
                            'close': data['close'][i],
                            'volume': data['volume'][i],
                            'timestamp': data.get('timestamp', [0]*length)[i]
                        })
                    
                    return {'5min': candles[-100:]}
                
                logger.warning(f"⚠️ Unexpected response format")
            else:
                logger.error(f"❌ API failed: {response.status_code}")
            
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
                
                logger.info(f"🔍 Fetching expiries (attempt {attempt+1}/{max_retries}) - Payload: {payload}")
                
                response = requests.post(
                    DHAN_EXPIRY_LIST_URL,
                    json=payload,
                    headers=self.headers,
                    timeout=10
                )
                
                logger.info(f"📡 Expiry API Status: {response.status_code}")
                
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
                    logger.error(f"❌ Expiry API failed: {response.status_code} - {response.text[:200]}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)
                        continue
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Expiry API timeout (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
            except Exception as e:
                logger.error(f"❌ Error getting expiry list: {e}")
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
            
            # Parse and filter future expiries
            future_expiries = []
            for e in expiry_list:
                try:
                    expiry_date = datetime.strptime(e, '%Y-%m-%d').date()
                    if expiry_date >= today:
                        future_expiries.append(expiry_date)
                except:
                    continue
            
            if not future_expiries:
                logger.warning(f"⚠️ {symbol}: No future expiries found in {expiry_list[:3]}")
                return None
            
            future_expiries.sort()
            
            if symbol_type == 'index':
                # Indices: Nearest weekly expiry
                selected = future_expiries[0]
                days_to_expiry = (selected - today).days
                logger.info(f"📅 {symbol}: Weekly expiry = {selected} ({days_to_expiry} days away)")
            else:
                # Stocks: Find monthly expiry (last Thursday pattern - day >= 20)
                monthly_expiries = [e for e in future_expiries if e.day >= 20]
                
                if monthly_expiries:
                    selected = monthly_expiries[0]
                    days_to_expiry = (selected - today).days
                    logger.info(f"📅 {symbol}: Monthly expiry = {selected} ({days_to_expiry} days away)")
                else:
                    # Fallback: Just use nearest expiry
                    selected = future_expiries[0]
                    days_to_expiry = (selected - today).days
                    logger.warning(f"⚠️ {symbol}: No monthly pattern found, using nearest = {selected} ({days_to_expiry} days)")
            
            return selected.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"❌ Error selecting expiry for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def update_expiry_for_symbol(self, symbol, security_id, segment, symbol_type):
        """Update expiry with auto-rollover logic"""
        try:
            # Get all expiries from API
            expiry_list = self.get_all_expiries(security_id, segment)
            
            if not expiry_list:
                logger.warning(f"⚠️ {symbol}: API returned no expiries")
                return None
            
            # Select best expiry from list
            selected_expiry = self.select_best_expiry(symbol, expiry_list, symbol_type)
            
            if not selected_expiry:
                logger.warning(f"⚠️ {symbol}: Could not select valid expiry from list: {expiry_list[:3]}")
                return None
            
            # Check if expiry changed (rollover detection)
            if symbol in self.expiry_map:
                old_expiry = self.expiry_map[symbol]
                if old_expiry != selected_expiry:
                    logger.warning(f"🔄 {symbol}: Expiry rollover! {old_expiry} → {selected_expiry}")
            
            # Update expiry map
            self.expiry_map[symbol] = selected_expiry
            return selected_expiry
            
        except Exception as e:
            logger.error(f"❌ Error updating expiry for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            
            logger.info(f"📡 Option Chain Request - Payload: {payload}")
            
            response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            self.last_option_chain_call = time.time()
            
            logger.info(f"📦 Option Chain Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"📊 Response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                
                if data.get('data'):
                    logger.info(f"✅ Option chain data received")
                    return data['data']
                else:
                    logger.warning(f"⚠️ No 'data' key in response: {data}")
            else:
                logger.error(f"❌ Option Chain API failed: {response.status_code} - {response.text[:200]}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting option chain: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def calculate_technical_indicators(self, candles):
        """Calculate key technical indicators"""
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            tr_list = []
            for i in range(1, min(15, len(candles))):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr = max(high_low, high_close, low_close)
                tr_list.append(tr)
            
            atr = sum(tr_list) / len(tr_list) if tr_list else 0
            price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            return {
                "current_price": closes[-1],
                "support": support,
                "resistance": resistance,
                "atr": atr,
                "price_change_pct": price_change_pct,
                "volume_spike": volume_spike,
                "avg_volume": avg_volume
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return None
    
    def analyze_option_chain_advanced(self, oc_data, spot_price, symbol):
        """Advanced option chain analysis with OI tracking"""
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
            
            # Track OI changes
            timestamp = datetime.now().isoformat()
            
            if symbol not in self.oi_history:
                self.oi_history[symbol] = deque(maxlen=5)
            
            self.oi_history[symbol].append({
                'timestamp': timestamp,
                'ce_oi': total_ce_oi,
                'pe_oi': total_pe_oi,
                'strikes': strike_wise_data
            })
            
            # Calculate OI change
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
    
    async def get_simple_ai_analysis(self, symbol, candles, technical_data, option_data, spot_price):
        """Simplified GPT analysis"""
        try:
            recent_candles = candles[-15:]
            candles_summary = [
                {
                    "open": c.get('open'),
                    "high": c.get('high'),
                    "low": c.get('low'),
                    "close": c.get('close'),
                    "volume": c.get('volume')
                }
                for c in recent_candles
            ]
            
            prompt = f"""Expert option trader analyzing {symbol}. Provide trading signal:

**Spot Price:** ₹{spot_price:,.2f}

**5-Min Technical (Last 100 candles):**
- Support: ₹{technical_data['support']:,.2f}
- Resistance: ₹{technical_data['resistance']:,.2f}
- ATR: ₹{technical_data['atr']:.2f}
- Price Change: {technical_data['price_change_pct']:.2f}%
- Volume Spike: {technical_data['volume_spike']:.2f}x

**Recent 15 Candles:**
{json.dumps(candles_summary, indent=2)}

**Option Chain:**
- PCR: {option_data['pcr']:.2f}
- ATM Strike: ₹{option_data['atm_strike']:,.0f}
- Max CE OI Strike: ₹{option_data.get('max_ce_oi_strike', 0):,.0f}
- Max PE OI Strike: ₹{option_data.get('max_pe_oi_strike', 0):,.0f}
- CE OI: {option_data['ce_total_oi']:,} | PE OI: {option_data['pe_total_oi']:,}
- ATM CE: ₹{option_data['atm_ce_price']:.2f} (IV: {option_data['atm_ce_iv']:.1f}%)
- ATM PE: ₹{option_data['atm_pe_price']:.2f} (IV: {option_data['atm_pe_iv']:.1f}%)

**Task:** Respond in JSON:
{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 0-100,
    "entry_price": price,
    "stop_loss": price,
    "target": price,
    "strike": strike_price,
    "reasoning": "brief (2-3 lines)",
    "risk_reward": ratio
}}

**Rules:**
- Signal only if confidence ≥ 70%
- Min R:R 1:2.5
- Consider price action + OI data
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Expert option trader. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            signal_data = json.loads(ai_response)
            
            logger.info(f"🤖 AI Signal: {signal_data.get('signal')} | Confidence: {signal_data.get('confidence')}%")
            
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return None
    
    def format_signal_message(self, symbol, signal_data, spot_price, expiry, technical_data):
        """Format trading signal for Telegram"""
        try:
            signal_type = signal_data.get('signal')
            
            if signal_type == "NO_TRADE":
                return None
            
            confidence = signal_data.get('confidence', 0)
            signal_emoji = "🟢 BUY CALL" if signal_type == "BUY_CE" else "🔴 BUY PUT"
            
            msg = f"{signal_emoji}\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"*{symbol}*\n"
            msg += f"Spot: ₹{spot_price:,.2f}\n"
            msg += f"Expiry: {expiry}\n\n"
            
            msg += f"*💰 Trade Setup:*\n"
            msg += f"Strike: ₹{signal_data.get('strike', 0):,.0f}\n"
            msg += f"Entry: ₹{signal_data.get('entry_price', 0):.2f}\n"
            msg += f"SL: ₹{signal_data.get('stop_loss', 0):.2f}\n"
            msg += f"Target: ₹{signal_data.get('target', 0):.2f}\n"
            msg += f"R:R = 1:{signal_data.get('risk_reward', 0):.2f}\n\n"
            
            msg += f"*🎯 Confidence:* {confidence}%\n\n"
            
            msg += f"*📍 Key Levels:*\n"
            msg += f"Support: ₹{technical_data['support']:,.2f}\n"
            msg += f"Resistance: ₹{technical_data['resistance']:,.2f}\n\n"
            
            msg += f"*💡 Analysis:*\n_{signal_data.get('reasoning', 'N/A')}_\n\n"
            
            msg += f"*⚠️ Risk:* 2-3% capital | Exit at SL\n\n"
            
            msg += f"🕒 {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
            
            return msg
            
        except Exception as e:
            logger.error(f"❌ Error formatting signal: {e}")
            return None
    
    async def analyze_and_send_signals(self, symbols_batch):
        """Enhanced analysis with API-based expiry"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                symbol_type = info['type']
                
                logger.info(f"📊 Analyzing {symbol} ({symbol_type})...")
                
                # Use actual API expiry
                expiry = self.update_expiry_for_symbol(symbol, security_id, segment, symbol_type)
                if not expiry:
                    logger.warning(f"⚠️ {symbol}: Could not determine expiry, skipping...")
                    await asyncio.sleep(2)
                    continue
                logger.info(f"📅 {symbol}: Using expiry: {expiry}")
                
                # Small delay after expiry API call
                await asyncio.sleep(1)
                
                # Get 5-min candle data
                candle_data = await self.get_multi_timeframe_data(security_id, segment)
                if not candle_data or '5min' not in candle_data:
                    logger.warning(f"⚠️ {symbol}: No candle data")
                    continue
                
                candles_5min = candle_data['5min']
                if len(candles_5min) < 50:
                    logger.warning(f"⚠️ {symbol}: Insufficient candles ({len(candles_5min)})")
                    continue
                
                logger.info(f"✅ Got {len(candles_5min)} candles")
                
                # Technical indicators
                technical_data = self.calculate_technical_indicators(candles_5min)
                if not technical_data:
                    continue
                
                spot_price = technical_data['current_price']
                logger.info(f"📈 Technical analysis done. Spot: ₹{spot_price:,.2f}")
                
                # Option chain with rate limit
                logger.info(f"🔍 Fetching option chain for {symbol} (Expiry: {expiry})...")
                oc_data = await self.get_option_chain_safe(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"⚠️ {symbol}: No option chain data")
                    await asyncio.sleep(3)
                    continue
                
                logger.info(f"✅ Option chain fetched")
                
                # Advanced OI analysis
                option_analysis = self.analyze_option_chain_advanced(oc_data, spot_price, symbol)
                if not option_analysis:
                    continue
                
                logger.info(f"📊 OI Analysis: PCR={option_analysis['pcr']:.2f} | OI Change={option_analysis['oi_change_pct']:+.2f}% | Snapshots={option_analysis['oi_snapshots']}")
                
                # GPT analysis
                signal_data = await self.get_simple_ai_analysis(
                    symbol, candles_5min, technical_data, option_analysis, spot_price
                )
                
                if not signal_data:
                    logger.warning(f"⚠️ {symbol}: AI analysis failed")
                    await asyncio.sleep(3)
                    continue
                
                # Send signal if confidence >= 70%
                if signal_data.get('signal') != 'NO_TRADE' and signal_data.get('confidence', 0) >= 70:
                    message = self.format_signal_message(symbol, signal_data, spot_price, expiry, technical_data)
                    if message:
                        await self.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"🚀 Signal sent for {symbol}!")
                else:
                    logger.info(f"⏸️ {symbol}: No actionable signal (Confidence: {signal_data.get('confidence', 0)}%)")
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    def is_market_hours(self):
        """Check if market is open (9:15 AM - 3:30 PM, Mon-Fri)"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Time check
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def send_startup_message(self):
        """Enhanced startup message"""
        try:
            indices_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'index'])
            stocks_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'stock'])
            
            msg = "🚀 *Advanced AI Option Trading Bot Started!*\n\n"
            msg += "🤖 *Powered by GPT-4o-mini*\n\n"
            
            msg += "*📊 Coverage:*\n"
            msg += f"• {indices_count} Indices (Weekly expiry)\n"
            msg += f"• {stocks_count} Stocks (Monthly expiry)\n\n"
            
            msg += "*🎯 Features:*\n"
            msg += "✅ 5-Min Timeframe Analysis\n"
            msg += "✅ Auto Expiry Selection (API-based)\n"
            msg += "✅ OI Change Tracking (5 snapshots)\n"
            msg += "✅ Rate Limit Protection\n"
            msg += "✅ PCR + Greeks Analysis\n\n"
            
            msg += "*⚙️ Settings:*\n"
            msg += "• Cycle: Every 5 minutes\n"
            msg += "• Min Confidence: 70%\n"
            msg += "• Min R:R: 1:2.5\n"
            msg += "• Position Size: 2-3% capital\n\n"
            
            msg += "*📅 Expiry Logic:*\n"
            msg += "• Indices: Nearest weekly\n"
            msg += "• Stocks: Nearest monthly (day >= 20)\n\n"
            
            msg += "⚠️ *Disclaimer:* Educational purposes only. Trade at your own risk.\n\n"
            msg += "_Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)_"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"❌ Error sending startup message: {e}")
    
    async def perform_analysis_cycle(self, indices, stocks, cycle_num):
        """Separate function for analysis cycle"""
        try:
            # Market time check
            if not self.is_market_hours():
                logger.warning("⚠️ Market is CLOSED! Running in test mode with limited data...")
            
            # Analyze indices (every cycle)
            if indices:
                logger.info(f"📊 Analyzing {len(indices)} indices...")
                await self.analyze_and_send_signals(indices)
                await asyncio.sleep(5)
            
            # Analyze stocks (batch processing)
            if stocks:
                logger.info(f"📈 Scanning {len(stocks)} stocks...")
                
                batch_size = 5
                stock_batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
                
                for batch_num, batch in enumerate(stock_batches, 1):
                    logger.info(f"📦 Stock Batch {batch_num}/{len(stock_batches)}: {batch}")
                    await self.analyze_and_send_signals(batch)
                    
                    if batch_num < len(stock_batches):
                        await asyncio.sleep(10)
            
            logger.info(f"✅ Cycle #{cycle_num} completed!")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {e}")
            raise
    
    async def run(self):
        """Main bot loop with intelligent batching"""
        logger.info("🚀 Advanced AI Option Trading Bot starting...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs. Exiting...")
            return
        
        await self.send_startup_message()
        
        # Separate indices and stocks
        indices = [s for s, info in self.security_id_map.items() if info['type'] == 'index']
        stocks = [s for s, info in self.security_id_map.items() if info['type'] == 'stock']
        
        logger.info(f"📊 {len(indices)} Indices | {len(stocks)} Stocks")
        
        # IMMEDIATE FIRST SCAN
        logger.info("🔥 IMMEDIATE SCAN: Starting first analysis cycle...")
        await self.perform_analysis_cycle(indices, stocks, cycle_num=1)
        logger.info("✅ Initial scan completed!")
        
        cycle_count = 1
        
        while self.running:
            try:
                cycle_count += 1
                
                logger.info(f"⏳ Waiting 5 minutes before next cycle...")
                await asyncio.sleep(300)
                
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"🔄 Cycle #{cycle_count} started at {timestamp}")
                
                await self.perform_analysis_cycle(indices, stocks, cycle_count)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)


# ========================
# RUN BOT
# ========================
if __name__ == "__main__":
    try:
        required_vars = [
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
            DHAN_CLIENT_ID,
            DHAN_ACCESS_TOKEN,
            OPENAI_API_KEY
        ]
        
        if not all(required_vars):
            logger.error("❌ Missing environment variables!")
            logger.error("Required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, OPENAI_API_KEY")
            exit(1)
        
        bot = AIOptionTradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        exit(1)
