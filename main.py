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

# Config from env
DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')
TELE_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELE_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL') or 60)

REQUIRED = [DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELE_TOKEN, TELE_CHAT_ID]

app = Flask(__name__)

def tele_send_http(chat_id: str, text: str):
    try:
        token = TELE_TOKEN
        if not token:
            logger.error('TELEGRAM_BOT_TOKEN not set')
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.warning('Telegram API returned %s: %s', r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.exception('Failed to send Telegram message: %s', e)
        return False

def get_nifty_expiry():
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')

def get_banknifty_expiry():
    today = datetime.now()
    days_ahead = 2 - today.weekday()  # Wednesday is 2
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')

def parse_expiry_string(expiry_str):
    if not expiry_str:
        return None
    expiry_str = expiry_str.strip()
    fmts = [
        '%d-%b-%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%d-%b-%y', '%d %b %Y',
        '%Y-%m-%d %H:%M:%S', '%d-%b-%Y %H:%M:%S'
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(expiry_str, fmt).date()
        except Exception:
            continue
    # try pattern like '07Oct2025' or '07Oct2025 00:00:00'
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
    """Download and filter only NIFTY / BANKNIFTY Index Options (OPTIDX)."""
    try:
        logger.info("📥 Downloading instruments from DhanHQ CSV...")
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        import csv
        from io import StringIO
        reader = csv.DictReader(StringIO(resp.text))

        fno_instruments = []
        for inst in reader:
            try:
                instr_name = (inst.get("SEM_INSTRUMENT_NAME") or '').upper()
                symbol = (inst.get("SEM_TRADING_SYMBOL") or '').upper()
                secid = inst.get("SEM_SMST_SECURITY_ID") or ''
                expiry = inst.get("SEM_EXPIRY_DATE") or ''
                strike = inst.get("SEM_STRIKE_PRICE") or ''
                opt_type = (inst.get("SEM_OPTION_TYPE") or '').upper()

                # Filter: instrument name must be OPTIDX and symbol must contain NIFTY or BANKNIFTY
                if instr_name == "OPTIDX" and ("NIFTY" in symbol or "BANKNIFTY" in symbol.replace(" ", "")):
                    # Normalize strike
                    try:
                        strike_val = float(strike)
                    except Exception:
                        # sometimes strike is '-0.01000' for FUT/others; skip non-numeric
                        continue

                    # expiry parsing to ensure valid
                    if not parse_expiry_string(expiry):
                        continue

                    fno_instruments.append({
                        "SEM_TRADING_SYMBOL": symbol,
                        "SEM_EXPIRY_DATE": expiry,
                        "SEM_STRIKE_PRICE": strike_val,
                        "SEM_OPTION_TYPE": opt_type,
                        "SEM_SMST_SECURITY_ID": secid
                    })
            except Exception:
                continue

        logger.info(f"✅ Filtered {len(fno_instruments)} index option instruments")
        if len(fno_instruments) == 0:
            # For debugging: sample some symbols
            resp2 = requests.get(url, timeout=60)
            reader2 = csv.DictReader(StringIO(resp2.text))
            sample = []
            for i, r in enumerate(reader2):
                if i >= 50: break
                sample.append(r.get("SEM_TRADING_SYMBOL") or r.get("SM_SYMBOL_NAME") or '')
            logger.warning("Sample symbols (first 50): %s", sample[:20])

        return fno_instruments

    except Exception as e:
        logger.exception("❌ Error fetching instruments: %s", e)
        return []

def get_spot_price_dhan(dhan, index_name):
    try:
        # Known mapping (may vary by account) — update if needed
        security_id = 13 if index_name == 'NIFTY' else 25
        exchange_seg = getattr(dhan, 'NSE', None) or 'NSE'
        response = dhan.market_quote(security_id=str(security_id), exchange_segment=exchange_seg)
        if response and response.get('status') in ('success', True):
            data = response.get('data', {})
            ltp_val = data.get('LTP') or data.get('ltp') or data.get('last_price') or 0
            ltp = float(ltp_val) if ltp_val not in (None, '') else 0.0
            logger.info("✅ %s Spot: ₹%s", index_name, f"{ltp:,.2f}")
            return ltp
        else:
            logger.error("Failed to get spot price: %s", response)
            return None
    except Exception as e:
        logger.exception("❌ Error getting spot price: %s", e)
        return None

def find_option_contracts(instruments, index_name, expiry_date, spot_price):
    try:
        logger.info("🔍 Finding options for %s, expiry: %s", index_name, expiry_date)
        if index_name == 'NIFTY':
            strike_gap = 50
            underlying_token = 'NIFTY'
        else:
            strike_gap = 100
            underlying_token = 'BANKNIFTY'

        atm = round(spot_price / strike_gap) * strike_gap
        strikes = [atm + (i * strike_gap) for i in range(-5, 6)]
        logger.info("🎯 ATM: %s, Strikes: %s..%s", atm, min(strikes), max(strikes))

        option_contracts = []
        try:
            target_dt = datetime.strptime(expiry_date, '%Y-%m-%d').date()
        except Exception:
            target_dt = parse_expiry_string(expiry_date)

        if target_dt is None:
            logger.warning("Target expiry couldn't be parsed: %s", expiry_date)
            return []

        for inst in instruments:
            try:
                sym = (inst.get('SEM_TRADING_SYMBOL') or '').upper()
                inst_expiry_raw = inst.get('SEM_EXPIRY_DATE', '')
                exp_dt = parse_expiry_string(inst_expiry_raw)
                if exp_dt is None:
                    continue
                if exp_dt != target_dt:
                    continue

                strike_val = inst.get('SEM_STRIKE_PRICE')
                try:
                    strike_val = float(strike_val)
                except Exception:
                    continue

                if strike_val in strikes and underlying_token in sym:
                    option_contracts.append({
                        'strike': strike_val,
                        'type': inst.get('SEM_OPTION_TYPE') or '',
                        'security_id': inst.get('SEM_SMST_SECURITY_ID') or '',
                        'symbol': inst.get('SEM_TRADING_SYMBOL'),
                        'expiry': inst_expiry_raw
                    })
            except Exception:
                continue

        logger.info("✅ Found %d option contracts", len(option_contracts))
        return sorted(option_contracts, key=lambda x: (x['strike'], x['type']))

    except Exception as e:
        logger.exception("❌ Error finding option contracts: %s", e)
        return []

def get_option_chain_data(dhan, option_contracts):
    try:
        if not option_contracts:
            return {}
        logger.info("📡 Fetching data for %d options...", len(option_contracts))
        result = {}
        exchange_seg = getattr(dhan, 'NSE_FNO', None) or getattr(dhan, 'FNO', None) or 'NSE_FNO'

        for opt in option_contracts[:40]:
            try:
                sec_id = str(opt['security_id'])
                if not sec_id:
                    continue
                response = dhan.market_quote(security_id=sec_id, exchange_segment=exchange_seg)
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
                time.sleep(0.08)
            except Exception as e:
                logger.warning("Failed to get quote for %s: %s", opt.get('security_id'), e)
                continue

        logger.info("✅ Fetched data for %d options", len(result))
        return result
    except Exception as e:
        logger.exception("❌ Error fetching option data: %s", e)
        return {}

def format_option_chain_message(index_name, spot_price, expiry, option_contracts, market_data):
    messages = []
    messages.append(f"📊 <b>{index_name}</b>")
    messages.append(f"💰 Spot: <b>₹{spot_price:,.2f}</b> | 📅 {expiry}\n")
    messages.append("<code>═══ CALL ═══╬══STRIKE══╬═══ PUT ═══</code>")
    messages.append("<code>  LTP    OI  ║  Price  ║  LTP    OI </code>")
    messages.append("─" * 42)

    strikes = {}
    for opt in option_contracts:
        strike = opt['strike']
        if strike not in strikes:
            strikes[strike] = {'CE': None, 'PE': None}
        sec_id = str(opt['security_id'])
        data = market_data.get(sec_id, {})
        strikes[strike][opt['type']] = data

    total_ce_oi = total_pe_oi = total_ce_vol = total_pe_vol = 0

    for strike in sorted(strikes.keys()):
        ce_data = strikes[strike].get('CE') or {}
        pe_data = strikes[strike].get('PE') or {}
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
        ce_oi_str = f"{ce_oi//1000}K" if ce_oi >= 1000 else f"{ce_oi}"
        pe_oi_str = f"{pe_oi//1000}K" if pe_oi >= 1000 else f"{pe_oi}"
        ce_str = f"{ce_ltp:>6.1f} {ce_oi_str:>5}" if ce_ltp > 0 else "   -      -  "
        pe_str = f"{pe_ltp:>6.1f} {pe_oi_str:>5}" if pe_ltp > 0 else "   -      -  "
        strike_str = f"{int(strike):>7}"
        messages.append(f"<code>{ce_str} ║ {strike_str} ║ {pe_str}</code>")

    messages.append("─" * 42)

    if total_ce_oi > 0 or total_pe_oi > 0:
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        messages.append(f"\n📊 <b>Open Interest:</b>")
        messages.append(f"   CALL: {total_ce_oi:>10,}")
        messages.append(f"   PUT:  {total_pe_oi:>10,}")
        messages.append(f"   <b>PCR: {pcr:.3f}</b>")
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
            logger.info(f"🔄 Iteration #{iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*50}")

            # NIFTY
            logger.info("--- Processing NIFTY ---")
            nifty_price = get_spot_price_dhan(dhan, 'NIFTY')
            if nifty_price and nifty_price > 0:
                nifty_expiry = get_nifty_expiry()
                nifty_options = find_option_contracts(instruments, 'NIFTY', nifty_expiry, nifty_price)
                if nifty_options:
                    market_data = get_option_chain_data(dhan, nifty_options)
                    if market_data:
                        msg = format_option_chain_message('NIFTY 50', nifty_price, nifty_expiry, nifty_options, market_data)
                        tele_send_http(TELE_CHAT_ID, msg)
                        logger.info("✅ NIFTY data sent to Telegram")
                        time.sleep(2)

            # BANKNIFTY
            logger.info("--- Processing BANKNIFTY ---")
            bn_price = get_spot_price_dhan(dhan, 'BANKNIFTY')
            if bn_price and bn_price > 0:
                bn_expiry = get_banknifty_expiry()
                bn_options = find_option_contracts(instruments, 'BANKNIFTY', bn_expiry, bn_price)
                if bn_options:
                    market_data = get_option_chain_data(dhan, bn_options)
                    if market_data:
                        msg = format_option_chain_message('BANK NIFTY', bn_price, bn_expiry, bn_options, market_data)
                        tele_send_http(TELE_CHAT_ID, msg)
                        logger.info("✅ BANKNIFTY data sent to Telegram")

            logger.info("✅ Iteration #%d complete. Sleeping %ds...", iteration, POLL_INTERVAL)

        except Exception as e:
            logger.exception("❌ Error in bot loop iteration #%d: %s", iteration, e)
            tele_send_http(TELE_CHAT_ID, f"⚠️ Error #{iteration}: {str(e)[:200]}")

        time.sleep(POLL_INTERVAL)

# start background thread
thread = threading.Thread(target=bot_loop, daemon=True)
thread.start()

@app.route('/')
def index():
    status = {
        'bot_thread_alive': thread.is_alive(),
        'poll_interval': POLL_INTERVAL,
        'service': 'DhanHQ Option Chain Bot v2.0',
        'features': ['Index Options (OPTIDX)', 'Real-time OI', 'Volume', 'PCR Analysis'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(status)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'thread_alive': thread.is_alive()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
