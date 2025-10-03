import os
import time
import threading
import logging
from flask import Flask, jsonify
import requests
from datetime import datetime, timedelta
from dhanhq import dhanhq

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('dhanhq-option-chain-bot')

# Load config from env
DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')
TELE_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELE_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL') or 60)

REQUIRED = [DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELE_TOKEN, TELE_CHAT_ID]

app = Flask(__name__)

# Exchange segment constants for DhanHQ 2.0+
IDX = 1  # Index segment
FNO = 2  # F&O segment

def tele_send_http(chat_id: str, text: str):
    """Send message using Telegram Bot HTTP API"""
    try:
        token = TELE_TOKEN
        if not token:
            logger.error('TELEGRAM_BOT_TOKEN not set')
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.warning('Telegram API returned %s: %s', r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.exception('Failed to send Telegram message: %s', e)
        return False

def get_nifty_expiry():
    """Get NIFTY 50 weekly expiry (next Thursday)"""
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')

def get_banknifty_expiry():
    """Get BANKNIFTY weekly expiry (next Wednesday)"""
    today = datetime.now()
    days_ahead = 2 - today.weekday()  # Wednesday is 2
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')

def get_instruments(dhan):
    """Download instruments master file from DhanHQ"""
    try:
        logger.info("📥 Fetching instruments from DhanHQ...")
        
        # Get instruments for F&O segment
        response = dhan.get_security_list(exchange_segment=FNO)
        
        if response and response.get('status') == 'success':
            instruments = response.get('data', [])
            logger.info(f"✅ Downloaded {len(instruments)} F&O instruments")
            return instruments
        else:
            logger.error(f"Failed to get instruments: {response}")
            return []
            
    except Exception as e:
        logger.exception(f"❌ Error fetching instruments: {e}")
        return []

def get_spot_price_dhan(dhan, index_name):
    """Get spot price for NIFTY or BANKNIFTY"""
    try:
        # DhanHQ Index security IDs
        # NIFTY 50 = 13, BANKNIFTY = 25
        security_id = "13" if index_name == 'NIFTY' else "25"
        
        # Fetch LTP using quote API
        response = dhan.get_ltp_data(
            exchange_segment=IDX,
            security_id=security_id
        )
        
        if response and response.get('status') == 'success':
            data = response.get('data', {})
            ltp = data.get('LTP', 0)
            logger.info(f"✅ {index_name} Spot: ₹{ltp:,.2f}")
            return ltp
        else:
            logger.error(f"Failed to get spot price: {response}")
            return None
            
    except Exception as e:
        logger.exception(f"❌ Error getting spot price: {e}")
        return None

def find_option_contracts(instruments, index_name, expiry_date, spot_price):
    """Find option contracts for strikes around spot price"""
    try:
        logger.info(f"🔍 Finding options for {index_name}, expiry: {expiry_date}")
        
        # Calculate strikes
        if index_name == 'NIFTY':
            strike_gap = 50
            underlying = 'NIFTY'
        else:
            strike_gap = 100
            underlying = 'BANKNIFTY'
        
        atm = round(spot_price / strike_gap) * strike_gap
        strikes = []
        for i in range(-5, 6):
            strikes.append(atm + (i * strike_gap))
        
        logger.info(f"🎯 ATM: {atm}, Strikes: {min(strikes)} to {max(strikes)}")
        
        # Filter instruments
        option_contracts = []
        for inst in instruments:
            if inst.get('SEM_TRADING_SYMBOL', '').startswith(underlying):
                inst_expiry = inst.get('SEM_EXPIRY_DATE', '')
                
                # Match expiry date (format: DD-MMM-YYYY)
                if inst_expiry:
                    try:
                        exp_dt = datetime.strptime(inst_expiry, '%d-%b-%Y')
                        target_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                        
                        if exp_dt.date() == target_dt.date():
                            strike = float(inst.get('SEM_STRIKE_PRICE', 0))
                            
                            if strike in strikes:
                                option_type = inst.get('SEM_OPTION_TYPE', '')
                                security_id = inst.get('SEM_SMST_SECURITY_ID')
                                
                                option_contracts.append({
                                    'strike': strike,
                                    'type': option_type,  # 'CE' or 'PE'
                                    'security_id': security_id,
                                    'symbol': inst.get('SEM_TRADING_SYMBOL'),
                                    'expiry': inst_expiry
                                })
                    except:
                        continue
        
        logger.info(f"✅ Found {len(option_contracts)} option contracts")
        return sorted(option_contracts, key=lambda x: (x['strike'], x['type']))
        
    except Exception as e:
        logger.exception(f"❌ Error finding option contracts: {e}")
        return []

def get_option_chain_data(dhan, option_contracts):
    """Fetch option chain data with OI and Volume"""
    try:
        if not option_contracts:
            return {}
        
        logger.info(f"📡 Fetching data for {len(option_contracts)} options...")
        
        # Prepare security IDs for batch quote
        security_ids = [str(opt['security_id']) for opt in option_contracts]
        
        # Fetch market quotes
        response = dhan.get_market_quote(
            exchange_segment=FNO,
            security_id=','.join(security_ids[:50])  # API limit: 50 at a time
        )
        
        if response and response.get('status') == 'success':
            data = response.get('data', {})
            logger.info(f"✅ Market data received")
            
            result = {}
            for quote in data.values():
                sec_id = str(quote.get('security_id'))
                result[sec_id] = {
                    'ltp': float(quote.get('LTP', 0)),
                    'oi': int(quote.get('open_interest', 0)),
                    'volume': int(quote.get('volume', 0)),
                    'change': float(quote.get('change', 0))
                }
            
            return result
        else:
            logger.error(f"Failed to get market quotes: {response}")
            return {}
            
    except Exception as e:
        logger.exception(f"❌ Error fetching option data: {e}")
        return {}

def calculate_strikes(spot_price, index_name, num_strikes=5):
    """Calculate ATM and surrounding strikes"""
    if index_name == 'NIFTY':
        strike_gap = 50
    else:
        strike_gap = 100
    
    atm = round(spot_price / strike_gap) * strike_gap
    strikes = []
    
    for i in range(-num_strikes, num_strikes + 1):
        strikes.append(atm + (i * strike_gap))
    
    return sorted(strikes)

def format_option_chain_message(index_name, spot_price, expiry, option_contracts, market_data):
    """Format option chain data for Telegram with OI and Volume"""
    messages = []
    messages.append(f"📊 <b>{index_name}</b>")
    messages.append(f"💰 Spot: <b>₹{spot_price:,.2f}</b> | 📅 {expiry}\n")
    
    # Header
    messages.append("<code>═══ CALL ═══╬══STRIKE══╬═══ PUT ═══</code>")
    messages.append("<code>  LTP    OI  ║  Price  ║  LTP    OI </code>")
    messages.append("─" * 42)
    
    # Group by strike
    strikes = {}
    for opt in option_contracts:
        strike = opt['strike']
        if strike not in strikes:
            strikes[strike] = {'CE': None, 'PE': None}
        
        sec_id = str(opt['security_id'])
        data = market_data.get(sec_id, {})
        
        strikes[strike][opt['type']] = data
    
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0
    
    for strike in sorted(strikes.keys()):
        ce_data = strikes[strike].get('CE', {}) or {}
        pe_data = strikes[strike].get('PE', {}) or {}
        
        ce_ltp = ce_data.get('ltp', 0)
        ce_oi = ce_data.get('oi', 0)
        ce_vol = ce_data.get('volume', 0)
        
        pe_ltp = pe_data.get('ltp', 0)
        pe_oi = pe_data.get('oi', 0)
        pe_vol = pe_data.get('volume', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        
        # Format OI in K
        ce_oi_str = f"{ce_oi//1000}K" if ce_oi >= 1000 else f"{ce_oi}"
        pe_oi_str = f"{pe_oi//1000}K" if pe_oi >= 1000 else f"{pe_oi}"
        
        # Format row
        ce_str = f"{ce_ltp:>6.1f} {ce_oi_str:>5}" if ce_ltp > 0 else "   -      -  "
        pe_str = f"{pe_ltp:>6.1f} {pe_oi_str:>5}" if pe_ltp > 0 else "   -      -  "
        strike_str = f"{int(strike):>7}"
        
        messages.append(f"<code>{ce_str} ║ {strike_str} ║ {pe_str}</code>")
    
    messages.append("─" * 42)
    
    # Summary
    if total_ce_oi > 0 or total_pe_oi > 0:
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        messages.append(f"\n📊 <b>Open Interest:</b>")
        messages.append(f"   CALL: {total_ce_oi:>10,}")
        messages.append(f"   PUT:  {total_pe_oi:>10,}")
        messages.append(f"   <b>PCR: {pcr:.3f}</b>")
        
        # PCR interpretation
        if pcr > 1.2:
            sentiment = "🟢 Bullish"
        elif pcr < 0.8:
            sentiment = "🔴 Bearish"
        else:
            sentiment = "🟡 Neutral"
        messages.append(f"   {sentiment}")
    
    if total_ce_vol > 0 or total_pe_vol > 0:
        messages.append(f"\n📈 <b>Volume:</b>")
        messages.append(f"   CALL: {total_ce_vol:>10,}")
        messages.append(f"   PUT:  {total_pe_vol:>10,}")
    
    messages.append(f"\n🕐 {time.strftime('%H:%M:%S')}")
    
    return "\n".join(messages)

def bot_loop():
    if not all(REQUIRED):
        logger.error('❌ Missing required environment variables')
        return

    try:
        # Initialize DhanHQ
        dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
        logger.info("✅ DhanHQ initialized successfully!")
    except Exception as e:
        logger.exception('❌ DhanHQ initialization failed: %s', e)
        tele_send_http(TELE_CHAT_ID, f'❌ DhanHQ init failed: {e}')
        return

    tele_send_http(TELE_CHAT_ID, 
                   f"✅ DhanHQ Option Chain Bot started!\n"
                   f"⏱ Polling every {POLL_INTERVAL}s\n"
                   f"📊 Real-time OI + Volume enabled!")
    
    # Download instruments once
    instruments = get_instruments(dhan)
    if not instruments:
        logger.error("❌ Failed to download instruments")
        tele_send_http(TELE_CHAT_ID, "❌ Failed to load instruments")
        return
    
    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Iteration #{iteration} - {time.strftime('%H:%M:%S')}")
            logger.info(f"{'='*50}")
            
            # Process NIFTY
            logger.info(f"\n--- Processing NIFTY ---")
            nifty_price = get_spot_price_dhan(dhan, 'NIFTY')
            
            if nifty_price and nifty_price > 0:
                nifty_expiry = get_nifty_expiry()
                nifty_options = find_option_contracts(instruments, 'NIFTY', 
                                                     nifty_expiry, nifty_price)
                
                if nifty_options:
                    market_data = get_option_chain_data(dhan, nifty_options)
                    if market_data:
                        msg = format_option_chain_message('NIFTY 50', nifty_price, 
                                                         nifty_expiry, nifty_options, 
                                                         market_data)
                        tele_send_http(TELE_CHAT_ID, msg)
                        logger.info("✅ NIFTY data sent to Telegram")
                        time.sleep(2)
            
            # Process BANKNIFTY
            logger.info(f"\n--- Processing BANKNIFTY ---")
            bn_price = get_spot_price_dhan(dhan, 'BANKNIFTY')
            
            if bn_price and bn_price > 0:
                bn_expiry = get_banknifty_expiry()
                bn_options = find_option_contracts(instruments, 'BANKNIFTY', 
                                                  bn_expiry, bn_price)
                
                if bn_options:
                    market_data = get_option_chain_data(dhan, bn_options)
                    if market_data:
                        msg = format_option_chain_message('BANK NIFTY', bn_price, 
                                                         bn_expiry, bn_options, 
                                                         market_data)
                        tele_send_http(TELE_CHAT_ID, msg)
                        logger.info("✅ BANKNIFTY data sent to Telegram")
            
            logger.info(f"✅ Iteration #{iteration} complete. Sleeping {POLL_INTERVAL}s...")
            
        except Exception as e:
            logger.exception(f"❌ Error in bot loop iteration #{iteration}: {e}")
            tele_send_http(TELE_CHAT_ID, f"⚠️ Error #{iteration}: {str(e)[:100]}")
        
        time.sleep(POLL_INTERVAL)

# Start bot in background thread
thread = threading.Thread(target=bot_loop, daemon=True)
thread.start()

@app.route('/')
def index():
    status = {
        'bot_thread_alive': thread.is_alive(),
        'poll_interval': POLL_INTERVAL,
        'service': 'DhanHQ Option Chain Bot v2.0',
        'features': ['Real-time OI', 'Volume', 'PCR Analysis'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(status)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'thread_alive': thread.is_alive()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
