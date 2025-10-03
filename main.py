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

def get_option_chain_dhan(dhan, index_name, expiry_date):
    """
    Fetch option chain data from DhanHQ
    index_name: 'NIFTY' or 'BANKNIFTY'
    expiry_date: 'YYYY-MM-DD' format
    """
    try:
        logger.info(f"📡 Fetching option chain for {index_name} expiry {expiry_date}")
        
        # DhanHQ security IDs for indices
        # NIFTY 50 = 13, BANKNIFTY = 25
        security_id = 13 if index_name == 'NIFTY' else 25
        
        # Fetch option chain
        response = dhan.get_option_chain(security_id, expiry_date)
        
        if response and response.get('status') == 'success':
            data = response.get('data', {})
            logger.info(f"✅ Option chain data received for {index_name}")
            return data
        else:
            logger.error(f"Failed to fetch option chain: {response}")
            return None
            
    except Exception as e:
        logger.exception(f"❌ Error fetching option chain: {e}")
        return None

def get_spot_price_dhan(dhan, index_name):
    """Get spot price for NIFTY or BANKNIFTY"""
    try:
        # DhanHQ Index security IDs
        security_id = 13 if index_name == 'NIFTY' else 25
        
        # Fetch LTP
        response = dhan.get_ltp_data(
            exchange_segment=dhan.IDX,
            security_id=str(security_id)
        )
        
        if response and response.get('status') == 'success':
            ltp = response['data']['LTP']
            logger.info(f"✅ {index_name} Spot: ₹{ltp:,.2f}")
            return ltp
        else:
            logger.error(f"Failed to get spot price: {response}")
            return None
            
    except Exception as e:
        logger.exception(f"❌ Error getting spot price: {e}")
        return None

def calculate_strikes(spot_price, index_name, num_strikes=5):
    """Calculate ATM and surrounding strikes"""
    if index_name == 'NIFTY':
        strike_gap = 50
    else:  # BANKNIFTY
        strike_gap = 100
    
    atm = round(spot_price / strike_gap) * strike_gap
    strikes = []
    
    for i in range(-num_strikes, num_strikes + 1):
        strikes.append(atm + (i * strike_gap))
    
    return sorted(strikes)

def format_option_chain_message(index_name, spot_price, expiry, option_data):
    """Format option chain data for Telegram with OI and Volume"""
    messages = []
    messages.append(f"📊 <b>{index_name}</b>")
    messages.append(f"💰 Spot: <b>₹{spot_price:,.2f}</b> | 📅 {expiry}\n")
    
    # Header
    messages.append("<code>═══ CALL ═══╬══STRIKE══╬═══ PUT ═══</code>")
    messages.append("<code>  LTP    OI  ║  Price  ║  LTP    OI </code>")
    messages.append("─" * 42)
    
    # Calculate strikes to display
    strikes = calculate_strikes(spot_price, index_name)
    
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0
    
    for strike in strikes:
        # Find CE and PE data for this strike
        ce_data = next((opt for opt in option_data.get('call', []) 
                       if opt.get('strike_price') == strike), None)
        pe_data = next((opt for opt in option_data.get('put', []) 
                       if opt.get('strike_price') == strike), None)
        
        # Extract values
        ce_ltp = ce_data.get('ltp', 0) if ce_data else 0
        ce_oi = ce_data.get('open_interest', 0) if ce_data else 0
        ce_vol = ce_data.get('volume', 0) if ce_data else 0
        
        pe_ltp = pe_data.get('ltp', 0) if pe_data else 0
        pe_oi = pe_data.get('open_interest', 0) if pe_data else 0
        pe_vol = pe_data.get('volume', 0) if pe_data else 0
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        
        # Format OI in K (thousands)
        ce_oi_str = f"{ce_oi//1000}K" if ce_oi >= 1000 else f"{ce_oi}"
        pe_oi_str = f"{pe_oi//1000}K" if pe_oi >= 1000 else f"{pe_oi}"
        
        # Format row
        ce_str = f"{ce_ltp:>6.1f} {ce_oi_str:>5}" if ce_ltp > 0 else "   -      -  "
        pe_str = f"{pe_ltp:>6.1f} {pe_oi_str:>5}" if pe_ltp > 0 else "   -      -  "
        strike_str = f"{int(strike):>7}"
        
        messages.append(f"<code>{ce_str} ║ {strike_str} ║ {pe_str}</code>")
    
    messages.append("─" * 42)
    
    # Summary with PCR
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
                   f"📊 Real-time OI + Volume data enabled!")
    
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
            
            if nifty_price:
                nifty_expiry = get_nifty_expiry()
                nifty_chain = get_option_chain_dhan(dhan, 'NIFTY', nifty_expiry)
                
                if nifty_chain:
                    msg = format_option_chain_message('NIFTY 50', nifty_price, 
                                                     nifty_expiry, nifty_chain)
                    tele_send_http(TELE_CHAT_ID, msg)
                    logger.info("✅ NIFTY data sent to Telegram")
                    time.sleep(2)
                else:
                    logger.warning("⚠️ No NIFTY option chain data")
            
            # Process BANKNIFTY
            logger.info(f"\n--- Processing BANKNIFTY ---")
            bn_price = get_spot_price_dhan(dhan, 'BANKNIFTY')
            
            if bn_price:
                bn_expiry = get_banknifty_expiry()
                bn_chain = get_option_chain_dhan(dhan, 'BANKNIFTY', bn_expiry)
                
                if bn_chain:
                    msg = format_option_chain_message('BANK NIFTY', bn_price, 
                                                     bn_expiry, bn_chain)
                    tele_send_http(TELE_CHAT_ID, msg)
                    logger.info("✅ BANKNIFTY data sent to Telegram")
                else:
                    logger.warning("⚠️ No BANKNIFTY option chain data")
            
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
        'service': 'DhanHQ Option Chain Bot',
        'features': ['Real-time OI', 'Volume', 'PCR Analysis'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(status)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'thread_alive': thread.is_alive()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
