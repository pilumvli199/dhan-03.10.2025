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
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I", "type": "index"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I", "type": "index"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I", "type": "index"},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "type": "index"},
    
    # High Volume Stocks (Monthly Expiry - 20 days before scan)
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
        self.expiry_map = {}  # Track expiries for each symbol
        self.oi_history = {}  # Track OI changes in memory (last 5 snapshots)
        self.last_option_chain_call = 0  # Rate limit tracking
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
        """
        🆕 Get candle data for multiple timeframes
        Returns: {5min, 15min, 1hour} data
        """
        try:
            timeframes = {
                '5': 90,   # Last 90 candles (5-min)
                '15': 60,  # Last 60 candles (15-min)
                '60': 30   # Last 30 candles (1-hour)
            }
            
            multi_tf_data = {}
            
            for interval, limit in timeframes.items():
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d 09:15:00")
                to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                payload = {
                    "securityId": str(security_id),
                    "exchangeSegment": segment,
                    "instrument": "EQUITY" if segment == "NSE_EQ" else "INDEX",
                    "interval": interval,
                    "fromDate": from_date,
                    "toDate": to_date
                }
                
                response = requests.post(
                    DHAN_INTRADAY_URL,
                    json=payload,
                    headers=self.headers,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle different response structures
                    if isinstance(data, dict):
                        if 'open' in data and isinstance(data['open'], list):
                            # Array format
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
                            multi_tf_data[f'{interval}min'] = candles[-limit:]
                        elif 'data' in data:
                            multi_tf_data[f'{interval}min'] = data['data'][-limit:]
                    
                    await asyncio.sleep(0.5)  # Small delay between TF requests
            
            return multi_tf_data if len(multi_tf_data) == 3 else None
            
        except Exception as e:
            logger.error(f"❌ Error getting multi-TF data: {e}")
            return None
    
    def get_all_expiries(self, security_id, segment):
        """
        🆕 Get ALL expiries for a symbol
        """
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            response = requests.post(
                DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    return data['data']
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting expiry list: {e}")
            return []
    
    def select_best_expiry(self, symbol, expiry_list, symbol_type):
        """
        🆕 Intelligent expiry selection
        - Indices: Weekly (nearest)
        - Stocks: Monthly (10 days before check)
        """
        try:
            if not expiry_list:
                return None
            
            today = datetime.now().date()
            future_expiries = [
                datetime.strptime(e, '%Y-%m-%d').date() 
                for e in expiry_list 
                if datetime.strptime(e, '%Y-%m-%d').date() >= today
            ]
            
            if not future_expiries:
                return None
            
            future_expiries.sort()
            
            if symbol_type == 'index':
                # Weekly expiry (nearest)
                selected = future_expiries[0]
                logger.info(f"📅 {symbol}: Weekly expiry selected: {selected}")
            else:
                # Stock: Monthly expiry (20 days before)
                for expiry in future_expiries:
                    days_to_expiry = (expiry - today).days
                    
                    # Check if within 20 days window
                    if 0 <= days_to_expiry <= 20:
                        selected = expiry
                        logger.info(f"📅 {symbol}: Within 20-day window! Expiry: {selected} ({days_to_expiry} days)")
                        break
                else:
                    # No expiry in window, skip this stock
                    logger.info(f"⏭️ {symbol}: No expiry within 20 days. Skipping...")
                    return None
            
            return selected.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"❌ Error selecting expiry: {e}")
            return None
    
    def update_expiry_for_symbol(self, symbol, security_id, segment, symbol_type):
        """
        🆕 Update expiry with auto-rollover logic
        """
        try:
            # Get all expiries
            expiry_list = self.get_all_expiries(security_id, segment)
            
            if not expiry_list:
                return None
            
            # Select best expiry
            selected_expiry = self.select_best_expiry(symbol, expiry_list, symbol_type)
            
            # Check if expiry changed
            if symbol in self.expiry_map:
                old_expiry = self.expiry_map[symbol]
                if old_expiry != selected_expiry:
                    logger.warning(f"🔄 {symbol}: Expiry rollover! {old_expiry} → {selected_expiry}")
            
            # Update expiry map
            if selected_expiry:
                self.expiry_map[symbol] = selected_expiry
            
            return selected_expiry
            
        except Exception as e:
            logger.error(f"❌ Error updating expiry: {e}")
            return None
    
    async def get_option_chain_safe(self, security_id, segment, expiry):
        """
        🆕 Rate-limit safe option chain fetch
        3 second minimum gap between calls
        """
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
                    return data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting option chain: {e}")
            return None
    
    def calculate_multi_tf_indicators(self, multi_tf_data):
        """
        🆕 Calculate technical indicators across timeframes
        """
        try:
            all_indicators = {}
            
            for tf, candles in multi_tf_data.items():
                if not candles or len(candles) < 20:
                    continue
                
                closes = [float(c['close']) for c in candles]
                highs = [float(c['high']) for c in candles]
                lows = [float(c['low']) for c in candles]
                volumes = [float(c['volume']) for c in candles]
                
                # Support/Resistance
                recent_highs = highs[-20:]
                recent_lows = lows[-20:]
                resistance = max(recent_highs)
                support = min(recent_lows)
                
                # ATR
                tr_list = []
                for i in range(1, min(15, len(candles))):
                    high_low = highs[i] - lows[i]
                    high_close = abs(highs[i] - closes[i-1])
                    low_close = abs(lows[i] - closes[i-1])
                    tr = max(high_low, high_close, low_close)
                    tr_list.append(tr)
                
                atr = sum(tr_list) / len(tr_list) if tr_list else 0
                
                # Trend
                price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
                
                # Volume
                avg_volume = sum(volumes[-20:]) / 20
                volume_spike = (volumes[-1] / avg_volume) if avg_volume > 0 else 1
                
                # Moving averages
                ma_20 = sum(closes[-20:]) / 20
                ma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma_20
                
                all_indicators[tf] = {
                    "current_price": closes[-1],
                    "support": support,
                    "resistance": resistance,
                    "atr": atr,
                    "price_change_pct": price_change,
                    "volume_spike": volume_spike,
                    "ma_20": ma_20,
                    "ma_50": ma_50,
                    "trend": "Bullish" if closes[-1] > ma_20 else "Bearish"
                }
            
            return all_indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating multi-TF indicators: {e}")
            return None
    
    def analyze_option_chain_advanced(self, oc_data, spot_price, symbol):
        """
        🆕 Advanced option chain analysis with OI change tracking
        """
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
            
            # 🆕 Track OI changes (last 5 snapshots in memory)
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
    
    async def get_advanced_ai_analysis(self, symbol, multi_tf_data, multi_tf_indicators, option_data, spot_price):
        """
        🆕 Enhanced GPT analysis with multi-TF context
        """
        try:
            # Prepare multi-TF summary
            tf_summary = ""
            for tf, indicators in multi_tf_indicators.items():
                tf_summary += f"\n**{tf.upper()} Timeframe:**\n"
                tf_summary += f"- Trend: {indicators['trend']}\n"
                tf_summary += f"- Price: ₹{indicators['current_price']:,.2f}\n"
                tf_summary += f"- Support: ₹{indicators['support']:,.2f}, Resistance: ₹{indicators['resistance']:,.2f}\n"
                tf_summary += f"- ATR: ₹{indicators['atr']:.2f}\n"
                tf_summary += f"- Price Change: {indicators['price_change_pct']:.2f}%\n"
                tf_summary += f"- Volume Spike: {indicators['volume_spike']:.2f}x\n"
            
            # Recent candles (5-min for entry timing)
            recent_candles_5m = multi_tf_data.get('5min', [])[-10:]
            candles_summary = []
            for c in recent_candles_5m:
                candles_summary.append({
                    "open": c.get('open'),
                    "high": c.get('high'),
                    "low": c.get('low'),
                    "close": c.get('close'),
                    "volume": c.get('volume')
                })
            
            prompt = f"""You are an expert option trader with multi-timeframe analysis expertise. Analyze {symbol}:

**SPOT PRICE:** ₹{spot_price:,.2f}

**MULTI-TIMEFRAME TECHNICAL ANALYSIS:**
{tf_summary}

**RECENT 10 CANDLES (5-min for precise entry):**
{json.dumps(candles_summary[-10:], indent=2)}

**OPTION CHAIN ANALYSIS:**
- PCR Ratio: {option_data['pcr']:.2f} {"(Bullish)" if option_data['pcr'] > 1.2 else "(Bearish)" if option_data['pcr'] < 0.8 else "(Neutral)"}
- ATM Strike: ₹{option_data['atm_strike']:,.0f}
- Max CE OI: ₹{option_data['max_ce_oi_strike']:,.0f} (Resistance)
- Max PE OI: ₹{option_data['max_pe_oi_strike']:,.0f} (Support)
- CE Total OI: {option_data['ce_total_oi']:,}
- PE Total OI: {option_data['pe_total_oi']:,}
- ATM CE: ₹{option_data['atm_ce_price']:.2f} (IV: {option_data['atm_ce_iv']:.1f}%)
- ATM PE: ₹{option_data['atm_pe_price']:.2f} (IV: {option_data['atm_pe_iv']:.1f}%)
- OI Change (last 5 snapshots): {option_data['oi_change_pct']:.2f}%

**ANALYSIS REQUIREMENTS:**
1. Check if ALL timeframes align (5min, 15min, 1hour)
2. Confirm with option chain (PCR, OI buildup)
3. Only give signal if confidence ≥ 70%
4. Minimum 1:2.5 risk-reward
5. Use ATR for stop loss calculation

**RESPOND IN THIS EXACT JSON FORMAT:**
{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 0-100,
    "entry_price": price,
    "stop_loss": price,
    "target": price,
    "strike": strike_price,
    "reasoning": "Multi-TF alignment + OI analysis summary (3-4 lines)",
    "risk_reward": ratio,
    "timeframe_alignment": "Strong/Moderate/Weak",
    "key_levels": {{"support": price, "resistance": price}}
}}

**CRITICAL:**
- NO TRADE if timeframes conflict
- Prefer lower TF for entry timing
- Higher TF for trend direction
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert multi-timeframe option trader. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            signal_data = json.loads(ai_response)
            
            logger.info(f"🤖 AI Signal for {symbol}: {signal_data.get('signal')} | Confidence: {signal_data.get('confidence')}% | TF Alignment: {signal_data.get('timeframe_alignment')}")
            
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return None
    
    def format_advanced_signal(self, symbol, signal_data, spot_price, expiry, multi_tf_indicators):
        """
        🆕 Enhanced signal formatting
        """
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
            
            msg += f"*📊 Multi-TF Analysis:*\n"
            for tf, ind in multi_tf_indicators.items():
                msg += f"{tf}: {ind['trend']} ({ind['price_change_pct']:+.2f}%)\n"
            msg += f"Alignment: {signal_data.get('timeframe_alignment', 'N/A')}\n\n"
            
            msg += f"*💰 Trade Setup:*\n"
            msg += f"Strike: ₹{signal_data.get('strike', 0):,.0f}\n"
            msg += f"Entry: ₹{signal_data.get('entry_price', 0):.2f}\n"
            msg += f"SL: ₹{signal_data.get('stop_loss', 0):.2f}\n"
            msg += f"Target: ₹{signal_data.get('target', 0):.2f}\n"
            msg += f"R:R = 1:{signal_data.get('risk_reward', 0):.2f}\n\n"
            
            msg += f"*🎯 Confidence:* {confidence}%\n\n"
            
            key_levels = signal_data.get('key_levels', {})
            if key_levels:
                msg += f"*📍 Key Levels:*\n"
                msg += f"Support: ₹{key_levels.get('support', 0):,.2f}\n"
                msg += f"Resistance: ₹{key_levels.get('resistance', 0):,.2f}\n\n"
            
            msg += f"*💡 Analysis:*\n_{signal_data.get('reasoning', 'N/A')}_\n\n"
            
            msg += f"*⚠️ Risk Management:*\n"
            msg += f"• Position size: 2-3% of capital\n"
            msg += f"• Exit at SL (No averaging)\n"
            msg += f"• Book 50% at 1:1.5, trail rest\n"
            msg += f"• No expiry day trading\n\n"
            
            msg += f"🕒 {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
            
            return msg
            
        except Exception as e:
            logger.error(f"❌ Error formatting signal: {e}")
            return None
    
    async def analyze_and_send_signals(self, symbols_batch):
        """
        🆕 Enhanced analysis with multi-TF + auto expiry
        """
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                symbol_type = info['type']
                
                logger.info(f"📊 Analyzing {symbol} ({symbol_type})...")
                
                # 🆕 Update expiry (auto-rollover + 20-day window for stocks)
                expiry = self.update_expiry_for_symbol(symbol, security_id, segment, symbol_type)
                
                if not expiry:
                    if symbol_type == 'stock':
                        logger.info(f"⏭️ {symbol}: Skipping (no expiry in 20-day window)")
                    continue
                
                # 🆕 Multi-timeframe data
                multi_tf_data = await self.get_multi_timeframe_data(security_id, segment)
                if not multi_tf_data or len(multi_tf_data) < 3:
                    logger.warning(f"⚠️ {symbol}: Insufficient multi-TF data")
                    continue
                
                logger.info(f"✅ Got data for {len(multi_tf_data)} timeframes")
                
                # 🆕 Multi-TF indicators
                multi_tf_indicators = self.calculate_multi_tf_indicators(multi_tf_data)
                if not multi_tf_indicators:
                    continue
                
                spot_price = multi_tf_indicators['5min']['current_price']
                logger.info(f"📈 Technical analysis complete. Spot: ₹{spot_price:,.2f}")
                
                # 🆕 Option chain with rate limit
                oc_data = await self.get_option_chain_safe(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"⚠️ {symbol}: No option chain data")
                    continue
                
                logger.info(f"✅ Option chain fetched (Expiry: {expiry})")
                
                # 🆕 Advanced OI analysis
                option_analysis = self.analyze_option_chain_advanced(oc_data, spot_price, symbol)
                if not option_analysis:
                    continue
                
                logger.info(f"📊 OI Analysis: PCR={option_analysis['pcr']:.2f} | OI Change={option_analysis['oi_change_pct']:+.2f}% | Snapshots={option_analysis['oi_snapshots']}")
                
                # 🆕 Advanced GPT analysis
                signal_data = await self.get_advanced_ai_analysis(
                    symbol, multi_tf_data, multi_tf_indicators, option_analysis, spot_price
                )
                
                if not signal_data:
                    logger.warning(f"⚠️ {symbol}: AI analysis failed")
                    continue
                
                # Send signal if confidence >= 70%
                if signal_data.get('signal') != 'NO_TRADE' and signal_data.get('confidence', 0) >= 70:
                    message = self.format_advanced_signal(
                        symbol, signal_data, spot_price, expiry, multi_tf_indicators
                    )
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
    
    async def run(self):
        """
        🆕 Main bot loop with intelligent batching
        """
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
        
        # 🆕 IMMEDIATE FIRST SCAN (No expiry wait!)
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
    
    async def perform_analysis_cycle(self, indices, stocks, cycle_num):
        """
        🆕 Separate function for analysis cycle (reusable for immediate scan)
        """
        try:
            # Analyze indices (every cycle)
            if indices:
                logger.info(f"📊 Analyzing {len(indices)} indices...")
                await self.analyze_and_send_signals(indices)
                await asyncio.sleep(5)
            
            # Analyze stocks (batch processing)
            if stocks:
                logger.info(f"📈 Scanning {len(stocks)} stocks (20-day expiry window)...")
                
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
    
    async def send_startup_message(self):
        """
        🆕 Enhanced startup message
        """
        try:
            indices_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'index'])
            stocks_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'stock'])
            
            msg = "🚀 *Advanced AI Option Trading Bot Started!*\n\n"
            msg += "🤖 *Powered by GPT-4o-mini*\n\n"
            
            msg += "*📊 Coverage:*\n"
            msg += f"• {indices_count} Indices (Weekly expiry)\n"
            msg += f"• {stocks_count} Stocks (Monthly expiry, 20-day window)\n\n"
            
            msg += "*🎯 Features:*\n"
            msg += "✅ Multi-Timeframe Analysis (5m, 15m, 1h)\n"
            msg += "✅ Auto Expiry Rollover\n"
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
            msg += "• Stocks: Monthly (20 days before)\n\n"
            
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
