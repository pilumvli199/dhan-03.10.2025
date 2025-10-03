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

# Exchange segment constants for DhanHQ 2.0+ (if library exposes constants, use them)
# We'll use dhanhq object's properties when calling API (dhan.NSE, dhan.NSE_FNO, etc.)

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

def parse_expiry_string(expiry_str):
    """Try multiple expiry date formats and return datetime.date or None."""
    from datetime import datetime
    if not expiry_str:
        return None
    expiry_str = expiry_str.strip()
    formats = ['%d-%b-%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%d-%b-%y', '%d %b %Y']
    for fmt in formats:
        try:
            return datetime.strptime(expiry_str, fmt).date()
        except Exception:
            continue
    # last-resort pattern like '07Oct2025'
    try:
        import re
        m = re.search(r'(\d{1,2}).*?([A-Za-z]{3}).*?(\d{4})', expiry_str)
        if m:
            ds = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            return datetime.strptime(ds, '%d-%b-%Y').date()
    except Exception:
        pass
    return None

def get_instruments(dhan):
    """Download and inspect instruments CSV; return filtered F&O option instruments (robust)."""
    try:
        logger.info("📥 Downloading instruments from DhanHQ API (robust loader)...")
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text

        # parse CSV and inspect header & first rows
        import csv
        from io import StringIO
        csv_data = StringIO(text)
        reader = csv.DictReader(csv_data)
        rows = []
        for i, r in enumerate(reader):
            if i < 10:
                rows.append(r)
            else:
                break

        if not rows:
            logger.warning("⚠️ CSV parsed but no rows found in preview.")
            return []

        # Log header keys and a sample row for debugging
        sample_keys = list(rows[0].keys())
        logger.debug("CSV header keys (sample): %s", sample_keys)
        logger.debug("CSV sample row 0 keys/values (trimmed): %s",
                     {k: rows[0].get(k) for k in sample_keys[:20]})

        # Heuristic: find likely column names for exchange, symbol, instr type, expiry, strike, security id
        def pick(col_candidates):
            for c in col_candidates:
                if c in sample_keys:
                    return c
            return None

        col_exchange = pick(['SEM_EXM_EXCH_ID', 'EXCHANGE_ID', 'EXCH_SEGMENT', 'EXCHANGE'])
        col_symbol = pick(['SEM_TRADING_SYMBOL', 'TRADING_SYMBOL', 'SYMBOL', 'SEM_SYMB'])
        col_instrument_name = pick(['SEM_INSTRUMENT_NAME', 'INSTRUMENT_NAME', 'INSTRUMENT'])
        col_expiry = pick(['SEM_EXPIRY_DATE', 'EXPIRY_DATE', 'EXPIRY'])
        col_strike = pick(['SEM_STRIKE_PRICE', 'STRIKE_PRICE', 'STRIKE'])
        col_option_type = pick(['SEM_OPTION_TYPE', 'OPTION_TYPE', 'OPT_TYPE'])
        col_security_id = pick(['SEM_SMST_SECURITY_ID', 'SMST_SECURITY_ID', 'SECURITY_ID', 'SMST_SECURITY_ID'])

        logger.debug("Detected columns -> exchange: %s, symbol: %s, instr_name: %s, expiry: %s, strike: %s, opt_type: %s, secid: %s",
                     col_exchange, col_symbol, col_instrument_name, col_expiry, col_strike, col_option_type, col_security_id)

        # Now re-iterate full CSV and filter robustly
        csv_data = StringIO(text)
        reader = csv.DictReader(csv_data)
        fno_instruments = []
        for inst in reader:
            try:
                symbol = inst.get(col_symbol, '') or inst.get('SEM_TRADING_SYMBOL', '') or ''
                instr_name = inst.get(col_instrument_name, '') or ''
                exch = inst.get(col_exchange, '') or ''
                # Normalize values to upper-case strings for safe checks
                symbol_u = symbol.upper()
                instr_name_u = instr_name.upper()
                exch_u = str(exch).strip()

                # Heuristics to identify F&O option indexes:
                # - instrument name contains 'OPT' (OPTIDX / OPTOBJ / OPT)
                # - exchange indicates F&O / NSE F&O (e.g., '2' in old CSVs) OR symbol contains 'NIFTY'/'BANKNIFTY'
                is_option = 'OPT' in instr_name_u or instr_name_u.startswith('OPTION') or 'OPTION' in instr_name_u
                is_index_underlying = ('NIFTY' in symbol_u) or ('BANKNIFTY' in symbol_u) or ('BANK NIFTY'.replace(' ', '') in symbol_u)
                is_fno_exchange = exch_u in ('2', 'FNO', 'NSE_FNO', 'NSE-FNO')

                # Accept if looks like an index option on NIFTY / BANKNIFTY
                if is_option and (is_index_underlying or is_fno_exchange):
                    # build normalized fields
                    expiry_raw = inst.get(col_expiry, '') or inst.get('SEM_EXPIRY_DATE', '') or ''
                    expiry_date = parse_expiry_string(expiry_raw)
                    strike = inst.get(col_strike, '') or inst.get('SEM_STRIKE_PRICE', '') or ''
                    try:
                        strike_val = float(str(strike)) if strike not in (None, '') else None
                    except Exception:
                        strike_val = None
                    option_type = (inst.get(col_option_type) or inst.get('SEM_OPTION_TYPE') or '').upper()
                    secid = inst.get(col_security_id) or inst.get('SEM_SMST_SECURITY_ID') or inst.get('SMST_SECURITY_ID') or ''

                    # add if we have strike and expiry and symbol contains NIFTY/BANKNIFTY
                    if (strike_val is not None) and expiry_date and ('NIFTY' in symbol_u or 'BANK' in symbol_u):
                        fno_instruments.append({
                            'SEM_TRADING_SYMBOL': symbol,
                            'SEM_EXPIRY_DATE': expiry_raw,
                            'SEM_STRIKE_PRICE': strike_val,
                            'SEM_OPTION_TYPE': option_type,
                            'SEM_SMST_SECURITY_ID': secid
                        })
            except Exception:
                continue

        logger.info(f"✅ Found {len(fno_instruments)} F&O option instruments (robust filter).")
        # If still zero, log first 50 sample symbols to inspect
        if len(fno_instruments) == 0:
            sample_symbols = []
            csv_data = StringIO(text)
            reader = csv.DictReader(csv_data)
            for i, r in enumerate(reader):
                if i >= 50: break
                sample_symbols.append(r.get(col_symbol) or r.get('SEM_TRADING_SYMBOL') or '')
            logger.warning("Sample symbols (first 50) for inspection: %s", sample_symbols[:20])

        return fno_instruments

    except Exception as e:
        logger.exception(f"❌ Error fetching instruments (robust): {e}")
        return []

def get_spot_price_dhan(dhan, index_name):
    """Get spot price for NIFTY or BANKNIFTY"""
    try:
        # DhanHQ Index security IDs might vary; using previously known mapping
        # NIFTY 50 = 13, BANKNIFTY = 25 (these are examples; verify in your account)
        security_id = 13 if index_name == 'NIFTY' else 25

        # Fetch LTP using market quote API
        response = dhan.market_quote(
            security_id=str(security_id),
            exchange_segment=getattr(dhan, 'NSE', None) or 'NSE'
        )

        if response and response.get('status') in ('success', True):
            data = response.get('data', {})
            # Typical keys may be 'LTP' or 'ltp'
            ltp_val = data.get('LTP') or data.get('ltp') or data.get('last_price') or 0
            ltp = float(ltp_val) if ltp_val not in (None, '') else 0.0
            logger.info(f"✅ {index_name} Spot: ₹{ltp:,.2f}")
            return ltp
        else:
            logger.error(f"Failed to get spot price: {response}")
            return None

    except Exception as e:
        logger.exception(f"❌ Error getting spot price: {e}")
        return None

def find_option_contracts(instruments, index_name, expiry_date, spot_price):
    """Find option contracts for strikes around spot price (robust expiry parsing)"""
    try:
        logger.info(f"🔍 Finding options for {index_name}, expiry: {expiry_date}")

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
        target_dt = None
        try:
            target_dt = datetime.strptime(expiry_date, '%Y-%m-%d').date()
        except Exception:
            logger.debug("Target expiry parse failed for %s", expiry_date)
            # fallback: try parse_expiry_string on given expiry_date string
            try:
                target_dt = parse_expiry_string(expiry_date)
            except Exception:
                target_dt = None

        for inst in instruments:
            try:
                sym = (inst.get('SEM_TRADING_SYMBOL') or '').upper()
                inst_expiry_raw = inst.get('SEM_EXPIRY_DATE', '')
                exp_dt = parse_expiry_string(inst_expiry_raw)
                if exp_dt is None or target_dt is None:
                    continue

                # match expiry
                if exp_dt != target_dt:
                    continue

                strike = inst.get('SEM_STRIKE_PRICE')
                try:
                    strike_val = float(strike)
                except Exception:
                    continue

                if strike_val in strikes and underlying in sym:
                    option_type = inst.get('SEM_OPTION_TYPE') or ''
                    secid = inst.get('SEM_SMST_SECURITY_ID') or ''
                    option_contracts.append({
                        'strike': strike_val,
                        'type': option_type,
                        'security_id': secid,
                        'symbol': inst.get('SEM_TRADING_SYMBOL'),
                        'expiry': inst_expiry_raw
                    })
            except Exception:
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

        result = {}

        # Fetch quotes in batches (API works better with individual calls)
        for opt in option_contracts[:40]:  # Increased limit for more strikes if needed
            try:
                sec_id = str(opt['security_id'])
                # use FNO exchange segment if available on the dhanhq object
                exchange_seg = getattr(dhan, 'NSE_FNO', None) or getattr(dhan, 'FNO', None) or 'NSE_FNO'
                response = dhan.market_quote(
                    security_id=sec_id,
                    exchange_segment=exchange_seg
                )

                if response and response.get('status') in ('success', True):
                    data = response.get('data', {})
                    ltp_val = data.get('LTP') or data.get('ltp') or data.get('last_price') or 0
                    oi_val = data.get('open_interest') or data.get('OI') or 0
                    vol_val = data.get('volume') or data.get('Volume') or 0
                    change_val = data.get('change') or data.get('Change') or 0
                    result[sec_id] = {
                        'ltp': float(ltp_val) if ltp_val not in (None, '') else 0.0,
                        'oi': int(oi_val) if oi_val not in (None, '') else 0,
                        'volume': int(vol_val) if vol_val not in (None, '') else 0,
                        'change': float(change_val) if change_val not in (None, '') else 0.0
                    }

                time.sleep(0.08)  # Small delay between requests

            except Exception as e:
                logger.warning(f"Failed to get quote for {sec_id}: {e}")
                continue

        logger.info(f"✅ Fetched data for {len(result)} options")
        return result

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
            tele_send_http(TELE_CHAT_ID, f"⚠️ Error #{iteration}: {str(e)[:200]}")

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
