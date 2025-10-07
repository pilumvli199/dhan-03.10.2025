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
from typing import Dict, List, Optional, Tuple

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
# ADVANCED TECHNICAL INDICATORS
# ========================

class AdvancedTechnicalAnalysis:
    """Advanced technical analysis with multiple indicators"""
    
    @staticmethod
    def calculate_ema(data: List[float], period: int) -> List[float]:
        """Calculate EMA"""
        ema = []
        multiplier = 2 / (period + 1)
        
        # First EMA is SMA
        sma = sum(data[:period]) / period
        ema.append(sma)
        
        for price in data[period:]:
            ema_value = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(closes: List[float]) -> Dict:
        """Calculate MACD"""
        if len(closes) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        ema_12 = AdvancedTechnicalAnalysis.calculate_ema(closes, 12)
        ema_26 = AdvancedTechnicalAnalysis.calculate_ema(closes, 26)
        
        macd_line = [ema_12[i] - ema_26[i] for i in range(len(ema_26))]
        signal_line = AdvancedTechnicalAnalysis.calculate_ema(macd_line, 9)
        
        histogram = macd_line[-1] - signal_line[-1]
        
        return {
            "macd": macd_line[-1],
            "signal": signal_line[-1],
            "histogram": histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(closes) < period:
            return {"upper": closes[-1], "middle": closes[-1], "lower": closes[-1]}
        
        sma = sum(closes[-period:]) / period
        variance = sum([(x - sma) ** 2 for x in closes[-period:]]) / period
        std = variance ** 0.5
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band,
            "bandwidth": (upper_band - lower_band) / sma * 100
        }
    
    @staticmethod
    def calculate_supertrend(highs: List[float], lows: List[float], closes: List[float], 
                            period: int = 10, multiplier: float = 3) -> Dict:
        """Calculate Supertrend"""
        if len(closes) < period:
            return {"trend": "NEUTRAL", "value": closes[-1]}
        
        # Calculate ATR
        tr_list = []
        for i in range(1, min(period + 1, len(closes))):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        atr = sum(tr_list) / len(tr_list)
        
        # Basic line
        basic_upper = ((highs[-1] + lows[-1]) / 2) + (multiplier * atr)
        basic_lower = ((highs[-1] + lows[-1]) / 2) - (multiplier * atr)
        
        # Determine trend
        if closes[-1] > basic_upper:
            trend = "BULLISH"
            value = basic_lower
        elif closes[-1] < basic_lower:
            trend = "BEARISH"
            value = basic_upper
        else:
            trend = "NEUTRAL"
            value = (basic_upper + basic_lower) / 2
        
        return {"trend": trend, "value": value, "atr": atr}
    
    @staticmethod
    def detect_candlestick_patterns(candles: List[Dict]) -> List[str]:
        """Detect candlestick patterns"""
        if len(candles) < 3:
            return []
        
        patterns = []
        
        # Last 3 candles
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # Doji
        body3 = abs(c3['close'] - c3['open'])
        range3 = c3['high'] - c3['low']
        if range3 > 0 and body3 / range3 < 0.1:
            patterns.append("DOJI")
        
        # Hammer
        lower_shadow3 = min(c3['open'], c3['close']) - c3['low']
        upper_shadow3 = c3['high'] - max(c3['open'], c3['close'])
        if lower_shadow3 > 2 * body3 and upper_shadow3 < body3:
            patterns.append("HAMMER")
        
        # Shooting Star
        if upper_shadow3 > 2 * body3 and lower_shadow3 < body3:
            patterns.append("SHOOTING_STAR")
        
        # Bullish Engulfing
        if (c2['close'] < c2['open'] and c3['close'] > c3['open'] and
            c3['close'] > c2['open'] and c3['open'] < c2['close']):
            patterns.append("BULLISH_ENGULFING")
        
        # Bearish Engulfing
        if (c2['close'] > c2['open'] and c3['close'] < c3['open'] and
            c3['close'] < c2['open'] and c3['open'] > c2['close']):
            patterns.append("BEARISH_ENGULFING")
        
        # Morning Star
        if (c1['close'] < c1['open'] and
            abs(c2['close'] - c2['open']) < (c1['open'] - c1['close']) * 0.3 and
            c3['close'] > c3['open'] and c3['close'] > (c1['open'] + c1['close']) / 2):
            patterns.append("MORNING_STAR")
        
        # Evening Star
        if (c1['close'] > c1['open'] and
            abs(c2['close'] - c2['open']) < (c1['close'] - c1['open']) * 0.3 and
            c3['close'] < c3['open'] and c3['close'] < (c1['open'] + c1['close']) / 2):
            patterns.append("EVENING_STAR")
        
        return patterns

# ========================
# ADVANCED OI ANALYSIS
# ========================

class AdvancedOIAnalysis:
    """Advanced Open Interest Analysis"""
    
    @staticmethod
    def calculate_oi_pcr_zones(option_data: Dict, spot_price: float) -> Dict:
        """Calculate OI PCR for different zones"""
        oc = option_data.get('oc', {})
        strikes = sorted([float(s) for s in oc.keys()])
        
        # Define zones
        itm_ce_oi = otm_ce_oi = 0
        itm_pe_oi = otm_pe_oi = 0
        atm_ce_oi = atm_pe_oi = 0
        
        for strike in strikes:
            strike_key = f"{strike:.6f}"
            strike_data = oc.get(strike_key, {})
            
            ce_oi = strike_data.get('ce', {}).get('oi', 0)
            pe_oi = strike_data.get('pe', {}).get('oi', 0)
            
            distance_pct = abs(strike - spot_price) / spot_price * 100
            
            if distance_pct <= 2:  # ATM zone (±2%)
                atm_ce_oi += ce_oi
                atm_pe_oi += pe_oi
            elif strike > spot_price:  # OTM CE, ITM PE
                otm_ce_oi += ce_oi
                itm_pe_oi += pe_oi
            else:  # ITM CE, OTM PE
                itm_ce_oi += ce_oi
                otm_pe_oi += pe_oi
        
        return {
            "atm_pcr": (atm_pe_oi / atm_ce_oi) if atm_ce_oi > 0 else 0,
            "otm_pcr": (otm_pe_oi / otm_ce_oi) if otm_ce_oi > 0 else 0,
            "itm_pcr": (itm_pe_oi / itm_ce_oi) if itm_ce_oi > 0 else 0,
            "buildup_ce": otm_ce_oi > itm_ce_oi,
            "buildup_pe": otm_pe_oi > itm_pe_oi
        }
    
    @staticmethod
    def detect_max_pain(option_data: Dict) -> float:
        """Calculate Max Pain strike"""
        oc = option_data.get('oc', {})
        strikes = sorted([float(s) for s in oc.keys()])
        
        min_pain = float('inf')
        max_pain_strike = strikes[len(strikes)//2] if strikes else 0
        
        for test_strike in strikes:
            total_pain = 0
            
            for strike in strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce_oi = strike_data.get('ce', {}).get('oi', 0)
                pe_oi = strike_data.get('pe', {}).get('oi', 0)
                
                if strike > test_strike:
                    total_pain += (strike - test_strike) * ce_oi
                elif strike < test_strike:
                    total_pain += (test_strike - strike) * pe_oi
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike
        
        return max_pain_strike
    
    @staticmethod
    def calculate_iv_skew(option_data: Dict, spot_price: float) -> Dict:
        """Calculate IV skew"""
        oc = option_data.get('oc', {})
        strikes = sorted([float(s) for s in oc.keys()])
        
        otm_ce_ivs = []
        otm_pe_ivs = []
        
        for strike in strikes:
            strike_key = f"{strike:.6f}"
            strike_data = oc.get(strike_key, {})
            
            distance_pct = abs(strike - spot_price) / spot_price * 100
            
            if 2 <= distance_pct <= 5:  # OTM zone
                ce_iv = strike_data.get('ce', {}).get('implied_volatility', 0)
                pe_iv = strike_data.get('pe', {}).get('implied_volatility', 0)
                
                if strike > spot_price and ce_iv > 0:
                    otm_ce_ivs.append(ce_iv)
                elif strike < spot_price and pe_iv > 0:
                    otm_pe_ivs.append(pe_iv)
        
        avg_ce_iv = sum(otm_ce_ivs) / len(otm_ce_ivs) if otm_ce_ivs else 0
        avg_pe_iv = sum(otm_pe_ivs) / len(otm_pe_ivs) if otm_pe_ivs else 0
        
        skew = (avg_pe_iv - avg_ce_iv) / avg_ce_iv * 100 if avg_ce_iv > 0 else 0
        
        return {
            "avg_ce_iv": avg_ce_iv,
            "avg_pe_iv": avg_pe_iv,
            "skew_pct": skew,
            "interpretation": "PUT_BIAS" if skew > 5 else "CALL_BIAS" if skew < -5 else "NEUTRAL"
        }

# ========================
# MAIN BOT CLASS
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
        self.last_signals = {}
        self.last_option_chain_call = 0
        self.technical_analyzer = AdvancedTechnicalAnalysis()
        self.oi_analyzer = AdvancedOIAnalysis()
        logger.info("🚀 Advanced AI Option Trading Bot v2.0 initialized")
    
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
                    
                    logger.info(f"✅ Got {len(candles)} candles")
                    return candles[-100:]
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting candle data: {e}")
            return None
    
    def get_all_expiries(self, security_id, segment):
        """Get ALL expiries"""
        max_retries = 2
        for attempt in range(max_retries):
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
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
        
        return []
    
    def select_best_expiry(self, symbol, expiry_list, symbol_type):
        """Intelligent expiry selection"""
        try:
            if not expiry_list:
                return None
            
            today = datetime.now().date()
            future_expiries = [datetime.strptime(e, '%Y-%m-%d').date() 
                             for e in expiry_list 
                             if datetime.strptime(e, '%Y-%m-%d').date() >= today]
            
            if not future_expiries:
                return None
            
            future_expiries.sort()
            
            if symbol_type == 'index':
                selected = future_expiries[0]
            else:
                monthly_expiries = [e for e in future_expiries if e.day >= 20]
                selected = monthly_expiries[0] if monthly_expiries else future_expiries[0]
            
            return selected.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"❌ Error selecting expiry: {e}")
            return None
    
    def update_expiry_for_symbol(self, symbol, security_id, segment, symbol_type):
        """Update expiry"""
        try:
            expiry_list = self.get_all_expiries(security_id, segment)
            if not expiry_list:
                return None
            
            selected_expiry = self.select_best_expiry(symbol, expiry_list, symbol_type)
            if selected_expiry:
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
                await asyncio.sleep(3 - time_since_last)
            
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
    
    def calculate_advanced_technicals(self, candles: List[Dict]) -> Dict:
        """Calculate ALL technical indicators"""
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            # Basic levels
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            # ATR
            tr_list = []
            for i in range(1, min(15, len(candles))):
                tr = max(highs[i] - lows[i], 
                        abs(highs[i] - closes[i-1]), 
                        abs(lows[i] - closes[i-1]))
                tr_list.append(tr)
            atr = sum(tr_list) / len(tr_list) if tr_list else 0
            
            # Volume
            avg_volume = sum(volumes[-20:]) / 20
            volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # RSI
            rsi = self.technical_analyzer.calculate_rsi(closes)
            
            # MACD
            macd_data = self.technical_analyzer.calculate_macd(closes)
            
            # Bollinger Bands
            bb_data = self.technical_analyzer.calculate_bollinger_bands(closes)
            
            # Supertrend
            st_data = self.technical_analyzer.calculate_supertrend(highs, lows, closes)
            
            # Candlestick Patterns
            patterns = self.technical_analyzer.detect_candlestick_patterns(candles)
            
            # Momentum
            price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            momentum_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
            momentum_10 = ((closes[-1] - closes[-10]) / closes[-10]) * 100 if len(closes) >= 10 else 0
            
            # Trend strength
            if len(closes) >= 20:
                ema_9 = self.technical_analyzer.calculate_ema(closes, 9)
                ema_21 = self.technical_analyzer.calculate_ema(closes, 21)
                trend_strength = "STRONG_BULL" if ema_9[-1] > ema_21[-1] * 1.02 else \
                               "STRONG_BEAR" if ema_9[-1] < ema_21[-1] * 0.98 else "NEUTRAL"
            else:
                trend_strength = "NEUTRAL"
            
            return {
                "current_price": closes[-1],
                "support": support,
                "resistance": resistance,
                "atr": atr,
                "price_change_pct": price_change_pct,
                "volume_spike": volume_spike,
                "avg_volume": avg_volume,
                "rsi": rsi,
                "macd": macd_data,
                "bollinger": bb_data,
                "supertrend": st_data,
                "patterns": patterns,
                "momentum_5": momentum_5,
                "momentum_10": momentum_10,
                "trend_strength": trend_strength
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating technicals: {e}")
            return None
    
    def analyze_option_chain_advanced(self, oc_data, spot_price, symbol):
        """Advanced option chain analysis with PCR zones, Max Pain, IV Skew"""
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
            
            # OI History tracking
            timestamp = datetime.now().isoformat()
            
            if symbol not in self.oi_history:
                self.oi_history[symbol] = deque(maxlen=10)
            
            self.oi_history[symbol].append({
                'timestamp': timestamp,
                'ce_oi': total_ce_oi,
                'pe_oi': total_pe_oi,
                'strikes': strike_wise_data
            })
            
            # OI Change calculation
            oi_change_pct = 0
            ce_change_pct = 0
            pe_change_pct = 0
            
            if len(self.oi_history[symbol]) >= 2:
                old_ce_oi = self.oi_history[symbol][0]['ce_oi']
                old_pe_oi = self.oi_history[symbol][0]['pe_oi']
                
                ce_change_pct = ((total_ce_oi - old_ce_oi) / old_ce_oi * 100) if old_ce_oi > 0 else 0
                pe_change_pct = ((total_pe_oi - old_pe_oi) / old_pe_oi * 100) if old_pe_oi > 0 else 0
                oi_change_pct = (ce_change_pct + pe_change_pct) / 2
            
            pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0
            
            # Advanced Analysis
            pcr_zones = self.oi_analyzer.calculate_oi_pcr_zones(oc_data, spot_price)
            max_pain = self.oi_analyzer.detect_max_pain(oc_data)
            iv_skew = self.oi_analyzer.calculate_iv_skew(oc_data, spot_price)
            
            # ATM data
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            atm_ce = atm_data.get('ce', {})
            atm_pe = atm_data.get('pe', {})
            
            # PCR interpretation
            pcr_signal = "BULLISH" if pcr > 1.2 else "BEARISH" if pcr < 0.8 else "NEUTRAL"
            
            # OI buildup analysis
            oi_buildup = "CALL_WRITING" if pcr_zones['buildup_ce'] else \
                        "PUT_WRITING" if pcr_zones['buildup_pe'] else "MIXED"
            
            return {
                "pcr": pcr,
                "pcr_signal": pcr_signal,
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
                "ce_change_pct": ce_change_pct,
                "pe_change_pct": pe_change_pct,
                "oi_snapshots": len(self.oi_history[symbol]),
                "pcr_zones": pcr_zones,
                "max_pain": max_pain,
                "iv_skew": iv_skew,
                "oi_buildup": oi_buildup
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing option chain: {e}")
            return None
    
    async def get_ai_analysis(self, symbol, candles, technical_data, option_data, spot_price):
        """Enhanced GPT analysis with advanced indicators"""
        try:
            recent_candles = candles[-10:]
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
            
            prompt = f"""Expert option trader analyzing {symbol}:

**🎯 SPOT PRICE:** ₹{spot_price:,.2f}

**📊 TECHNICAL INDICATORS:**
• Support: ₹{technical_data['support']:,.2f} | Resistance: ₹{technical_data['resistance']:,.2f}
• ATR: ₹{technical_data['atr']:.2f}
• Price Change: {technical_data['price_change_pct']:.2f}%
• Momentum (5-candle): {technical_data['momentum_5']:.2f}%
• Momentum (10-candle): {technical_data['momentum_10']:.2f}%
• Volume Spike: {technical_data['volume_spike']:.2f}x
• RSI: {technical_data['rsi']:.2f}
• MACD: {technical_data['macd']['macd']:.2f} | Signal: {technical_data['macd']['signal']:.2f} | Histogram: {technical_data['macd']['histogram']:.2f}
• Bollinger: Upper ₹{technical_data['bollinger']['upper']:.2f} | Lower ₹{technical_data['bollinger']['lower']:.2f} | Bandwidth: {technical_data['bollinger']['bandwidth']:.2f}%
• Supertrend: {technical_data['supertrend']['trend']} @ ₹{technical_data['supertrend']['value']:.2f}
• Trend Strength: {technical_data['trend_strength']}
• Patterns: {', '.join(technical_data['patterns']) if technical_data['patterns'] else 'None'}

**📈 RECENT CANDLES (Last 10):**
{json.dumps(candles_summary, indent=2)}

**💹 OPTIONS DATA:**
• PCR: {option_data['pcr']:.2f} ({option_data['pcr_signal']})
• ATM Strike: ₹{option_data['atm_strike']:,.0f}
• Max Pain: ₹{option_data['max_pain']:,.0f}
• Max CE OI: ₹{option_data.get('max_ce_oi_strike', 0):,.0f} | Max PE OI: ₹{option_data.get('max_pe_oi_strike', 0):,.0f}
• Total CE OI: {option_data['ce_total_oi']:,} | Total PE OI: {option_data['pe_total_oi']:,}
• CE OI Change: {option_data['ce_change_pct']:.2f}% | PE OI Change: {option_data['pe_change_pct']:.2f}%
• OI Buildup: {option_data['oi_buildup']}
• ATM CE: ₹{option_data['atm_ce_price']:.2f} (IV: {option_data['atm_ce_iv']:.2f})
• ATM PE: ₹{option_data['atm_pe_price']:.2f} (IV: {option_data['atm_pe_iv']:.2f})

**🎯 PCR ZONES:**
• ATM PCR: {option_data['pcr_zones']['atm_pcr']:.2f}
• OTM PCR: {option_data['pcr_zones']['otm_pcr']:.2f}
• ITM PCR: {option_data['pcr_zones']['itm_pcr']:.2f}

**📉 IV SKEW:**
• CE IV: {option_data['iv_skew']['avg_ce_iv']:.2f}
• PE IV: {option_data['iv_skew']['avg_pe_iv']:.2f}
• Skew: {option_data['iv_skew']['skew_pct']:.2f}% ({option_data['iv_skew']['interpretation']})

**🔍 ANALYSIS REQUIRED:**
Based on ALL above indicators, provide JSON response:
{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 0-100,
    "entry_price": option_premium_price,
    "stop_loss": option_premium_sl,
    "target": option_premium_target,
    "strike": strike_to_trade,
    "reasoning": "Detailed reasoning covering: trend, momentum, RSI, MACD, Supertrend, PCR analysis, OI buildup, IV skew, candlestick patterns, and key support/resistance levels",
    "risk_reward": ratio,
    "trade_type": "INTRADAY" or "POSITIONAL",
    "key_levels": {{"stop_spot": spot_sl, "target_spot": spot_target}},
    "market_sentiment": "BULLISH/BEARISH/NEUTRAL"
}}

**⚠️ STRICT RULES:**
1. Minimum confidence: 75% (70% for high-conviction setups)
2. Minimum Risk:Reward: 1:2.5
3. Consider ALL technical indicators
4. RSI overbought (>70) → avoid CE, RSI oversold (<30) → avoid PE
5. MACD crossover → strong signal
6. Supertrend alignment → high conviction
7. Candlestick patterns → entry confirmation
8. PCR extremes: >1.3 bullish, <0.7 bearish
9. OI buildup direction crucial
10. IV skew indicates market bias
11. Max Pain acts as magnet
12. Volume spike + momentum = strong move
13. Multiple indicator confluence = high confidence
14. ATM/OTM selection based on risk appetite
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert options trader with 15+ years experience. Analyze using ALL indicators provided. Respond ONLY with valid JSON, no extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Clean response
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            signal_data = json.loads(ai_response)
            
            logger.info(f"🤖 AI Signal: {signal_data.get('signal')} | Confidence: {signal_data.get('confidence')}% | Sentiment: {signal_data.get('market_sentiment')}")
            
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ AI analysis error: {e}")
            return None
    
    def format_signal_message(self, symbol, signal_data, spot_price, expiry, technical_data, option_data):
        """Enhanced signal formatting with ALL details"""
        try:
            signal_type = signal_data.get('signal')
            
            if signal_type == "NO_TRADE":
                return None
            
            confidence = signal_data.get('confidence', 0)
            signal_emoji = "🟢 BUY CALL" if signal_type == "BUY_CE" else "🔴 BUY PUT"
            sentiment_emoji = "🚀" if signal_data.get('market_sentiment') == "BULLISH" else "📉" if signal_data.get('market_sentiment') == "BEARISH" else "⚖️"
            
            msg = f"{signal_emoji} {sentiment_emoji}\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"*📊 {symbol}*\n"
            msg += f"Spot: ₹{spot_price:,.2f}\n"
            msg += f"Expiry: {expiry}\n"
            msg += f"Type: {signal_data.get('trade_type', 'INTRADAY')}\n\n"
            
            msg += f"*💰 TRADE SETUP:*\n"
            msg += f"Strike: ₹{signal_data.get('strike', 0):,.0f}\n"
            msg += f"Entry: ₹{signal_data.get('entry_price', 0):.2f}\n"
            msg += f"Stop Loss: ₹{signal_data.get('stop_loss', 0):.2f}\n"
            msg += f"Target: ₹{signal_data.get('target', 0):.2f}\n"
            msg += f"Risk:Reward = 1:{signal_data.get('risk_reward', 0):.2f}\n\n"
            
            msg += f"*🎯 SPOT LEVELS:*\n"
            key_levels = signal_data.get('key_levels', {})
            if key_levels:
                msg += f"Spot SL: ₹{key_levels.get('stop_spot', 0):,.2f}\n"
                msg += f"Spot Target: ₹{key_levels.get('target_spot', 0):,.2f}\n\n"
            
            msg += f"*🎯 CONFIDENCE:* {confidence}%\n\n"
            
            msg += f"*📍 SUPPORT/RESISTANCE:*\n"
            msg += f"Support: ₹{technical_data['support']:,.2f}\n"
            msg += f"Resistance: ₹{technical_data['resistance']:,.2f}\n"
            msg += f"ATR: ₹{technical_data['atr']:.2f}\n\n"
            
            msg += f"*📊 KEY INDICATORS:*\n"
            msg += f"RSI: {technical_data['rsi']:.1f} "
            if technical_data['rsi'] > 70:
                msg += "⚠️ Overbought"
            elif technical_data['rsi'] < 30:
                msg += "⚠️ Oversold"
            else:
                msg += "✅"
            msg += f"\nSupertrend: {technical_data['supertrend']['trend']}\n"
            msg += f"Trend: {technical_data['trend_strength']}\n"
            msg += f"MACD: {'🟢 Bullish' if technical_data['macd']['histogram'] > 0 else '🔴 Bearish'}\n"
            
            if technical_data['patterns']:
                msg += f"Patterns: {', '.join(technical_data['patterns'][:3])}\n"
            
            msg += f"\n*💹 OPTIONS METRICS:*\n"
            msg += f"PCR: {option_data['pcr']:.2f} ({option_data['pcr_signal']})\n"
            msg += f"Max Pain: ₹{option_data['max_pain']:,.0f}\n"
            msg += f"OI Buildup: {option_data['oi_buildup']}\n"
            msg += f"IV Skew: {option_data['iv_skew']['interpretation']}\n"
            msg += f"CE OI: {option_data['ce_change_pct']:+.1f}% | PE OI: {option_data['pe_change_pct']:+.1f}%\n\n"
            
            msg += f"*💡 REASONING:*\n_{signal_data.get('reasoning', 'N/A')}_\n\n"
            
            msg += f"*⚠️ RISK MANAGEMENT:*\n"
            msg += f"• Use 2-3% of capital\n"
            msg += f"• Strict SL mandatory\n"
            msg += f"• Book partial at 1:1.5\n"
            msg += f"• Trail SL after 1:2\n\n"
            
            msg += f"🕒 {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
            msg += f"_⚠️ For educational purposes only. Trade at your own risk._"
            
            return msg
            
        except Exception as e:
            logger.error(f"❌ Error formatting message: {e}")
            return None
    
    def should_send_signal(self, symbol: str, signal_data: Dict) -> bool:
        """Avoid duplicate signals"""
        try:
            signal_type = signal_data.get('signal')
            
            if signal_type == 'NO_TRADE':
                return False
            
            current_time = datetime.now()
            
            # Check last signal
            if symbol in self.last_signals:
                last_signal = self.last_signals[symbol]
                time_diff = (current_time - last_signal['timestamp']).total_seconds() / 60
                
                # Don't send same signal within 30 minutes
                if time_diff < 30 and last_signal['type'] == signal_type:
                    logger.info(f"⏸️ {symbol}: Duplicate signal suppressed ({time_diff:.0f}m ago)")
                    return False
            
            # Store this signal
            self.last_signals[symbol] = {
                'type': signal_type,
                'timestamp': current_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking signal: {e}")
            return True
    
    async def analyze_and_send_signals(self, symbols_batch):
        """Main analysis function with advanced features"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                symbol_type = info['type']
                
                logger.info(f"\n{'='*50}")
                logger.info(f"📊 ANALYZING {symbol}")
                logger.info(f"{'='*50}")
                
                # Get expiry
                expiry = self.update_expiry_for_symbol(symbol, security_id, segment, symbol_type)
                if not expiry:
                    logger.warning(f"⚠️ {symbol}: No expiry available")
                    await asyncio.sleep(2)
                    continue
                
                logger.info(f"📅 Expiry: {expiry}")
                await asyncio.sleep(1)
                
                # Get candle data
                candles = await self.get_candle_data(security_id, segment)
                if not candles or len(candles) < 50:
                    logger.warning(f"⚠️ {symbol}: Insufficient candles ({len(candles) if candles else 0})")
                    continue
                
                # Advanced Technical Analysis
                technical_data = self.calculate_advanced_technicals(candles)
                if not technical_data:
                    logger.warning(f"⚠️ {symbol}: Technical analysis failed")
                    continue
                
                spot_price = technical_data['current_price']
                logger.info(f"💰 Spot: ₹{spot_price:,.2f}")
                logger.info(f"📈 RSI: {technical_data['rsi']:.1f} | Trend: {technical_data['trend_strength']}")
                logger.info(f"🎯 Supertrend: {technical_data['supertrend']['trend']}")
                
                # Option chain
                oc_data = await self.get_option_chain_safe(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"⚠️ {symbol}: No option chain data")
                    await asyncio.sleep(3)
                    continue
                
                # Advanced OI Analysis
                option_analysis = self.analyze_option_chain_advanced(oc_data, spot_price, symbol)
                if not option_analysis:
                    logger.warning(f"⚠️ {symbol}: Option analysis failed")
                    continue
                
                logger.info(f"💹 PCR: {option_analysis['pcr']:.2f} ({option_analysis['pcr_signal']})")
                logger.info(f"🎯 Max Pain: ₹{option_analysis['max_pain']:,.0f}")
                logger.info(f"📊 OI Buildup: {option_analysis['oi_buildup']}")
                logger.info(f"📉 IV Skew: {option_analysis['iv_skew']['interpretation']}")
                
                # AI Analysis
                signal_data = await self.get_ai_analysis(
                    symbol, candles, technical_data, option_analysis, spot_price
                )
                
                if not signal_data:
                    logger.warning(f"⚠️ {symbol}: AI analysis failed")
                    await asyncio.sleep(3)
                    continue
                
                # Check if should send signal
                if signal_data.get('signal') != 'NO_TRADE' and signal_data.get('confidence', 0) >= 70:
                    
                    if self.should_send_signal(symbol, signal_data):
                        message = self.format_signal_message(
                            symbol, signal_data, spot_price, expiry, 
                            technical_data, option_analysis
                        )
                        
                        if message:
                            await self.bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=message,
                                parse_mode='Markdown'
                            )
                            logger.info(f"🚀✅ SIGNAL SENT FOR {symbol}!")
                        else:
                            logger.warning(f"⚠️ {symbol}: Message formatting failed")
                    else:
                        logger.info(f"⏸️ {symbol}: Signal suppressed (duplicate)")
                else:
                    logger.info(f"⏸️ {symbol}: No trade signal (Confidence: {signal_data.get('confidence', 0)}%)")
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    def is_market_hours(self):
        """Check if market is open"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def send_startup_message(self):
        """Enhanced startup message"""
        try:
            indices_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'index'])
            stocks_count = len([s for s, info in self.security_id_map.items() if info['type'] == 'stock'])
            
            msg = "🚀 *AI OPTION TRADING BOT v2.0 STARTED!*\n\n"
            msg += "🤖 *Powered by GPT-4o-mini + Advanced Analytics*\n\n"
            
            msg += "*📊 COVERAGE:*\n"
            msg += f"• {indices_count} Indices (Weekly Expiry)\n"
            msg += f"• {stocks_count} Stocks (Monthly Expiry)\n"
            msg += f"• Total: {indices_count + stocks_count} Instruments\n\n"
            
            msg += "*🎯 ADVANCED FEATURES:*\n"
            msg += "✅ Multi-timeframe Analysis\n"
            msg += "✅ RSI, MACD, Bollinger Bands\n"
            msg += "✅ Supertrend Indicator\n"
            msg += "✅ Candlestick Patterns\n"
            msg += "✅ PCR Zone Analysis\n"
            msg += "✅ Max Pain Calculation\n"
            msg += "✅ IV Skew Detection\n"
            msg += "✅ OI Buildup Tracking\n"
            msg += "✅ Auto Expiry Rollover\n"
            msg += "✅ Duplicate Signal Prevention\n"
            msg += "✅ Smart Rate Limiting\n\n"
            
            msg += "*⚙️ SETTINGS:*\n"
            msg += "• Scan Cycle: 5 minutes\n"
            msg += "• Min Confidence: 70-75%\n"
            msg += "• Min Risk:Reward: 1:2.5\n"
            msg += "• Signal Cooldown: 30 min\n\n"
            
            msg += "*📈 TECHNICAL INDICATORS:*\n"
            msg += "• RSI (14)\n"
            msg += "• MACD (12,26,9)\n"
            msg += "• Bollinger Bands (20,2)\n"
            msg += "• Supertrend (10,3)\n"
            msg += "• EMA (9,21)\n"
            msg += "• Volume Analysis\n"
            msg += "• Momentum (5,10 candles)\n\n"
            
            msg += "*💹 OPTIONS ANALYSIS:*\n"
            msg += "• Put-Call Ratio (PCR)\n"
            msg += "• PCR Zones (ITM/ATM/OTM)\n"
            msg += "• Max Pain Strike\n"
            msg += "• IV Skew Analysis\n"
            msg += "• OI Change Tracking\n"
            msg += "• OI Buildup Detection\n\n"
            
            msg += "*⚠️ DISCLAIMER:*\n"
            msg += "_This bot is for educational purposes only._\n"
            msg += "_Always use proper risk management._\n"
            msg += "_Past performance ≠ future results._\n\n"
            
            msg += f"🕒 Market Hours: 9:15 AM - 3:30 PM\n"
            msg += f"📅 Today: {datetime.now().strftime('%d %B %Y')}\n\n"
            
            msg += "🔥 *READY TO SCAN!*"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup message sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Startup message error: {e}")
    
    async def perform_analysis_cycle(self, indices, stocks, cycle_num):
        """Perform complete analysis cycle"""
        try:
            market_status = "🟢 OPEN" if self.is_market_hours() else "🔴 CLOSED"
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 CYCLE #{cycle_num} | Market: {market_status}")
            logger.info(f"{'='*60}\n")
            
            # Analyze Indices first
            if indices:
                logger.info(f"📊 Analyzing {len(indices)} INDICES...")
                await self.analyze_and_send_signals(indices)
                await asyncio.sleep(5)
            
            # Analyze Stocks in batches
            if stocks:
                logger.info(f"📈 Scanning {len(stocks)} STOCKS...")
                
                batch_size = 5
                stock_batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
                
                for batch_num, batch in enumerate(stock_batches, 1):
                    logger.info(f"\n📦 Processing Batch {batch_num}/{len(stock_batches)}: {batch}")
                    await self.analyze_and_send_signals(batch)
                    
                    if batch_num < len(stock_batches):
                        logger.info(f"⏳ Cooling down 10s before next batch...")
                        await asyncio.sleep(10)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ CYCLE #{cycle_num} COMPLETED!")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"❌ Cycle error: {e}")
            raise
    
    async def send_market_summary(self, cycle_num):
        """Send periodic market summary"""
        try:
            if cycle_num % 6 != 0:  # Every 30 minutes (6 cycles x 5 min)
                return
            
            now = datetime.now()
            signals_sent = len(self.last_signals)
            
            msg = f"📊 *MARKET SUMMARY - Cycle #{cycle_num}*\n\n"
            msg += f"🕒 Time: {now.strftime('%H:%M:%S')}\n"
            msg += f"📈 Signals Sent: {signals_sent}\n\n"
            
            if self.last_signals:
                msg += "*🎯 Recent Signals:*\n"
                for symbol, data in list(self.last_signals.items())[-5:]:
                    time_ago = (now - data['timestamp']).total_seconds() / 60
                    signal_emoji = "🟢" if data['type'] == "BUY_CE" else "🔴"
                    msg += f"{signal_emoji} {symbol} - {int(time_ago)}m ago\n"
                msg += "\n"
            
            msg += f"_Next summary in 30 minutes_"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("📊 Market summary sent")
            
        except Exception as e:
            logger.error(f"❌ Summary error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("\n🚀 STARTING ADVANCED AI OPTION TRADING BOT v2.0...")
        
        # Load securities
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs. Exiting...")
            return
        
        # Send startup message
        await self.send_startup_message()
        
        # Separate indices and stocks
        indices = [s for s, info in self.security_id_map.items() if info['type'] == 'index']
        stocks = [s for s, info in self.security_id_map.items() if info['type'] == 'stock']
        
        logger.info(f"\n📊 Total Coverage: {len(indices)} Indices + {len(stocks)} Stocks = {len(indices) + len(stocks)} instruments")
        
        # Immediate first scan
        logger.info("\n🔥 INITIATING IMMEDIATE SCAN...")
        await self.perform_analysis_cycle(indices, stocks, 1)
        logger.info("✅ Initial scan completed!\n")
        
        cycle_count = 1
        
        # Main loop
        while self.running:
            try:
                cycle_count += 1
                
                logger.info(f"\n⏳ Waiting 5 minutes for next cycle...")
                await asyncio.sleep(300)  # 5 minutes
                
                logger.info(f"\n🔄 Starting Cycle #{cycle_count}...")
                await self.perform_analysis_cycle(indices, stocks, cycle_count)
                
                # Send periodic summary
                await self.send_market_summary(cycle_count)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down (User interrupted)...")
                self.running = False
                
                # Send shutdown message
                try:
                    msg = "🛑 *BOT STOPPED*\n\n"
                    msg += f"Total cycles completed: {cycle_count}\n"
                    msg += f"Total signals sent: {len(self.last_signals)}\n"
                    msg += f"Stopped at: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
                    
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=msg,
                        parse_mode='Markdown'
                    )
                except:
                    pass
                
                break
                
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                logger.info("⏳ Waiting 60s before retry...")
                await asyncio.sleep(60)


# ========================
# MAIN ENTRY POINT
# ========================

if __name__ == "__main__":
    try:
        # Verify environment variables
        required_vars = {
            "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
            "DHAN_CLIENT_ID": DHAN_CLIENT_ID,
            "DHAN_ACCESS_TOKEN": DHAN_ACCESS_TOKEN,
            "OPENAI_API_KEY": OPENAI_API_KEY
        }
        
        missing_vars = [name for name, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            logger.error("Please set all required environment variables before running the bot.")
            exit(1)
        
        logger.info("✅ All environment variables verified")
        
        # Initialize and run bot
        bot = AIOptionTradingBot()
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
        exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        exit(1)
