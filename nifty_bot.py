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

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if AI_PROVIDER == "groq":
    from groq import Groq
    ai_client = Groq(api_key=GROQ_API_KEY)
    AI_MODEL = "llama-3.1-70b-versatile"
    logger.info("Using Groq")
else:
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    AI_MODEL = "gpt-4o-mini"
    logger.info("Using OpenAI")

DHAN_API_BASE = "https://api.dhan.co"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

CANDLES_TO_FETCH = 200
TECHNICAL_LOOKBACK = 50
ATR_PERIOD = 30
AI_CANDLES_COUNT = 30

STOCKS_INDICES = {
    "NIFTY": {"symbol": "NIFTY 50", "segment": "IDX_I", "type": "index"},
    "BANKNIFTY": {"symbol": "NIFTY BANK", "segment": "IDX_I", "type": "index"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I", "type": "index"},
    "FINNIFTY": {"symbol": "FINNIFTY", "segment": "IDX_I", "type": "index"},
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
        logger.info("Bot initialized")
    
    async def load_security_ids(self):
        try:
            logger.info("Loading security IDs...")
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
                                        logger.info(f"{symbol}: {sec_id}")
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
                                        logger.info(f"{symbol}: {sec_id}")
                                        break
                        except Exception:
                            continue
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"Loaded {len(self.security_id_map)} securities")
                return True
            else:
                logger.error(f"Failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    async def get_candle_data(self, security_id, segment):
        try:
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
            
            response = requests.post(DHAN_INTRADAY_URL, json=payload, headers=self.headers, timeout=15)
            
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
                    
                    logger.info(f"Got {len(candles)} candles")
                    return candles[-CANDLES_TO_FETCH:]
                
            return None
            
        except Exception as e:
            logger.error(f"Candle error: {e}")
            return None
    
    def get_all_expiries(self, security_id, segment):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {"UnderlyingScrip": security_id, "UnderlyingSeg": segment}
                response = requests.post(DHAN_EXPIRY_LIST_URL, json=payload, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and data.get('data'):
                        return data['data']
                    else:
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)
                            continue
                elif response.status_code == 429:
                    import time
                    time.sleep(5)
                    if attempt < max_retries - 1:
                        continue
                
            except Exception as e:
                logger.error(f"Expiry error: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
        
        return []
    
    def select_best_expiry(self, symbol, expiry_list, symbol_type):
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
                return None
            
            future_expiries.sort()
            
            if symbol_type == 'index':
                selected = future_expiries[0]
            else:
                monthly_expiries = [e for e in future_expiries if e.day >= 20]
                selected = monthly_expiries[0] if monthly_expiries else future_expiries[0]
            
            return selected.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Expiry select error: {e}")
            return None
    
    def update_expiry_for_symbol(self, symbol, security_id, segment, symbol_type):
        try:
            expiry_list = self.get_all_expiries(security_id, segment)
            if not expiry_list:
                return None
            
            selected_expiry = self.select_best_expiry(symbol, expiry_list, symbol_type)
            if selected_expiry:
                self.expiry_map[symbol] = selected_expiry
            
            return selected_expiry
            
        except Exception as e:
            logger.error(f"Update expiry error: {e}")
            return None
    
    async def get_option_chain_safe(self, security_id, segment, expiry):
        try:
            import time
            current_time = time.time()
            time_since_last = current_time - self.last_option_chain_call
            
            if time_since_last < 3:
                await asyncio.sleep(3 - time_since_last)
            
            payload = {"UnderlyingScrip": security_id, "UnderlyingSeg": segment, "Expiry": expiry}
            response = requests.post(DHAN_OPTION_CHAIN_URL, json=payload, headers=self.headers, timeout=15)
            
            self.last_option_chain_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None
    
    def calculate_technical_indicators(self, candles):
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            resistance = max(highs[-TECHNICAL_LOOKBACK:])
            support = min(lows[-TECHNICAL_LOOKBACK:])
            
            tr_list = []
            atr_period = min(ATR_PERIOD, len(candles) - 1)
            
            for i in range(1, atr_period + 1):
                high_low = highs[-i] - lows[-i]
                high_close = abs(highs[-i] - closes[-i-1])
                low_close = abs(lows[-i] - closes[-i-1])
                tr_list.append(max(high_low, high_close, low_close))
            
            atr = sum(tr_list) / len(tr_list) if tr_list else 0
            price_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            avg_volume = sum(volumes[-50:]) / 50
            volume_spike = (volumes[-1] / avg_volume) if avg_volume > 0 else 1
            
            short_ma = sum(closes[-10:]) / 10
            long_ma = sum(closes[-30:]) / 30
            trend = "BULLISH" if short_ma > long_ma else "BEARISH"
            
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
            logger.error(f"Technical error: {e}")
            return None
    
    def analyze_option_chain_advanced(self, oc_data, spot_price, symbol):
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
            
            for strike in relevant_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                total_ce_oi += ce.get('oi', 0)
                total_pe_oi += pe.get('oi', 0)
                total_ce_volume += ce.get('volume', 0)
                total_pe_volume += pe.get('volume', 0)
            
            pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0
            
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            atm_ce = atm_data.get('ce', {})
            atm_pe = atm_data.get('pe', {})
            
            return {
                "pcr": pcr,
                "atm_strike": atm_strike,
                "ce_total_oi": total_ce_oi,
                "pe_total_oi": total_pe_oi,
                "ce_total_volume": total_ce_volume,
                "pe_total_volume": total_pe_volume,
                "atm_ce_price": atm_ce.get('last_price', 0),
                "atm_pe_price": atm_pe.get('last_price', 0),
                "atm_ce_iv": atm_ce.get('implied_volatility', 0),
                "atm_pe_iv": atm_pe.get('implied_volatility', 0)
            }
            
        except Exception as e:
            logger.error(f"OC analysis error: {e}")
            return None
    
    async def get_ai_analysis(self, symbol, candles, technical_data, option_data, spot_price):
        try:
            recent_candles = candles[-AI_CANDLES_COUNT:]
            
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
            
            atm_ce_premium = option_data.get('atm_ce_price', 0)
            atm_pe_premium = option_data.get('atm_pe_price', 0)
            
            prompt = f"""Expert option trader. Analyze {symbol} trading signal.

Market: Spot Rs{spot_price:.2f}
Trend: {technical_data['trend']}
Support: Rs{technical_data['support']:.2f}
Resistance: Rs{technical_data['resistance']:.2f}
ATR: Rs{technical_data['atr']:.2f}
PCR: {option_data['pcr']:.2f}

ATM Strike: Rs{option_data['atm_strike']:.0f}
CE Premium: Rs{atm_ce_premium:.2f}
PE Premium: Rs{atm_pe_premium:.2f}

Candles: {json.dumps(candles_summary[:5])}

Respond ONLY JSON:
{{
    "signal": "BUY_CE" or "BUY_PE" or "NO_TRADE",
    "confidence": 70-95,
    "strike": {option_data['atm_strike']},
    "entry_price": {atm_ce_premium:.2f},
    "stop_loss": {max(atm_ce_premium * 0.65, 0.5):.2f},
    "target": {atm_ce_premium * 1.8:.2f},
    "risk_reward": 2.5,
    "reasoning": "short reason"
}}

Use OPTION premiums only. SL < Entry < Target. RR >= 2.0"""

            response = ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": "Expert options trader. JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            analysis = json.loads(content)
            
            required = ["signal", "confidence", "strike", "entry_price", "stop_loss", "target", "risk_reward", "reasoning"]
            if not all(f in analysis for f in required):
                return None
            
            if analysis['signal'] in ['BUY_CE', 'BUY_PE']:
                if analysis['stop_loss'] >= analysis['entry_price'] or analysis['target'] <= analysis['entry_price'] or analysis['confidence'] < 65:
                    return None
            
            logger.info(f"AI: {analysis['signal']} @ Rs{analysis.get('entry_price', 0):.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"AI error: {e}")
            return None
    
    async def send_telegram_message(self, message):
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def analyze_symbol(self, symbol):
        try:
            logger.info(f"Analyzing: {symbol}")
            
            if symbol not in self.security_id_map:
                return
            
            sec_info = self.security_id_map[symbol]
            security_id = sec_info['security_id']
            segment = sec_info['segment']
            symbol_type = sec_info['type']
            
            candles = await self.get_candle_data(security_id, segment)
            if not candles or len(candles) < 50:
                return
            
            technical_data = self.calculate_technical_indicators(candles)
            if not technical_data:
                return
            
            spot_price = technical_data['current_price']
            logger.info(f"{symbol}: Rs{spot_price:.2f} {technical_data['trend']}")
            
            expiry = self.update_expiry_for_symbol(symbol, security_id, segment, symbol_type)
            if not expiry:
                return
            
            oc_data = await self.get_option_chain_safe(security_id, segment, expiry)
            if not oc_data:
                return
            
            option_data = self.analyze_option_chain_advanced(oc_data, spot_price, symbol)
            if not option_data:
                return
            
            logger.info(f"PCR: {option_data['pcr']:.2f}")
            
            ai_analysis = await self.get_ai_analysis(symbol, candles, technical_data, option_data, spot_price)
            
            if ai_analysis and ai_analysis['signal'] in ['BUY_CE', 'BUY_PE']:
                await self.send_trade_signal(symbol, ai_analysis, technical_data, option_data, expiry)
            else:
                logger.info(f"{symbol}: NO_TRADE - {ai_analysis.get('reasoning', 'No clear signal') if ai_analysis else 'Analysis failed'}")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
    
    async def send_trade_signal(self, symbol, ai, tech, opt, expiry):
        try:
            signal_type = "CALL" if ai['signal'] == 'BUY_CE' else "PUT"
            
            msg = f"""<b>{signal_type} - {symbol}</b>

Spot: Rs{tech['current_price']:.2f}
Trend: {tech['trend']}
PCR: {opt['pcr']:.2f}

Strike: Rs{ai['strike']:.0f}
Expiry: {expiry}
Entry: Rs{ai['entry_price']:.2f}
SL: Rs{ai['stop_loss']:.2f}
Target: Rs{ai['target']:.2f}
RR: {ai['risk_reward']:.2f}

Confidence: {ai['confidence']}%

{ai['reasoning']}

{datetime.now().strftime('%d-%m-%Y %I:%M %p')}"""
            
            await self.send_telegram_message(msg)
            logger.info(f"{symbol}: Signal sent")
            
        except Exception as e:
            logger.error(f"Signal error: {e}")
    
    async def run_analysis_cycle(self):
        try:
            logger.info(f"Cycle start: {datetime.now().strftime('%I:%M %p')}")
            
            for symbol in STOCKS_INDICES.keys():
                await self.analyze_symbol(symbol)
                await asyncio.sleep(2)
            
            logger.info("Cycle complete")
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
    
    async def start(self):
        try:
            success = await self.load_security_ids()
            if not success:
                logger.error("Failed to load IDs")
                return
            
            await self.send_telegram_message("<b>Bot Started</b>")
            logger.info("Starting main loop...")
            
            while self.running:
                try:
                    current_time = datetime.now()
                    logger.info(f"Current time: {current_time.strftime('%I:%M %p')}")
                    
                    if current_time.weekday() < 5:
                        market_start = current_time.replace(hour=9, minute=15, second=0)
                        market_end = current_time.replace(hour=15, minute=30, second=0)
                        
                        if market_start <= current_time <= market_end:
                            logger.info("Market hours - Starting analysis")
                            await self.run_analysis_cycle()
                            logger.info("Waiting 15 min")
                            await asyncio.sleep(900)
                        else:
                            logger.info("Outside market hours - Waiting 5 min")
                            await asyncio.sleep(300)
                    else:
                        logger.info("Weekend - Waiting 1 hour")
                        await asyncio.sleep(3600)
                
                except Exception as cycle_error:
                    logger.error(f"Cycle error: {cycle_error}")
                    await asyncio.sleep(60)
                    
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            try:
                await self.send_telegram_message("<b>Bot Stopped</b>")
            except:
                pass

async def main():
    bot = AIOptionTradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
