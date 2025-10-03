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

                if instr_name == "OPTIDX" and ("NIFTY" in symbol or "BANKNIFTY" in symbol.replace(" ", "")):
                    try:
                        strike_val = float(strike)
                    except Exception:
                        continue
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
            from io import StringIO
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

# ---------------- New helper: robust method discovery & call -----------------
def discover_dhan_methods(dhan):
    """Return callable names on the dhan object useful for market data."""
    try:
        methods = [m for m in dir(dhan) if callable(getattr(dhan, m))]
        # filter likely candidates by name
        candidates = [m for m in methods if any(x in m.lower() for x in ('quote','market','ltp','price','get'))]
        logger.debug("dhan callable candidates (len=%d): %s", len(candidates), candidates[:50])
        return candidates
    except Exception as e:
        logger.exception("discover_dhan_methods failed: %s", e)
        return []

def try_dhan_quote(dhan, security_id=None, symbol=None, exchange_segment=None):
    """
    Try a list of common method names and argument patterns on the dhan object
    to fetch a market quote. Return a normalized dict on success, else None.
    """
    try:
        # candidate method names to try (ordered)
        name_candidates = [
            'market_quote','get_market_quote','get_quote','quote','get_quotes','quotes',
            'getLTP','ltp','get_ltp','get_quotes_by_security','get_quote_by_security_id',
            'get_quote_by_symbol','fetch_quote','fetch_market_quote','getMarketData','getMarketQuote'
        ]

        # also include discovered methods from object (helps custom SDKs)
        discovered = discover_dhan_methods(dhan)
        for d in discovered:
            if d not in name_candidates:
                name_candidates.append(d)

        # try calling with different arg patterns
        for name in name_candidates:
            if not hasattr(dhan, name):
                continue
            method = getattr(dhan, name)
            # try keyword args first
            tried = []
            try:
                # pattern 1: security_id + exchange_segment
                if security_id is not None:
                    try:
                        resp = method(security_id=str(security_id), exchange_segment=exchange_segment)
                        tried.append((name, 'security_id, exchange_segment'))
                        if resp and isinstance(resp, dict) and any(k.lower() in str(resp).lower() for k in ('ltp','open_interest','status','data')):
                            logger.debug("try_dhan_quote: got response from %s with security_id", name)
                            return resp
                    except Exception:
                        pass
                # pattern 2: security_id only
                if security_id is not None:
                    try:
                        resp = method(str(security_id))
                        tried.append((name, 'security_id_pos'))
                        if resp and isinstance(resp, dict) and any(k.lower() in str(resp).lower() for k in ('ltp','open_interest','status','data')):
                            logger.debug("try_dhan_quote: got response from %s with sec_id positional", name)
                            return resp
                    except Exception:
                        pass
                # pattern 3: symbol
                if symbol is not None:
                    try:
                        resp = method(symbol)
                        tried.append((name, 'symbol_pos'))
                        if resp and isinstance(resp, dict) and any(k.lower() in str(resp).lower() for k in ('ltp','open_interest','status','data')):
                            logger.debug("try_dhan_quote: got response from %s with symbol_pos", name)
                            return resp
                    except Exception:
                        pass
                    try:
                        resp = method(symbol=symbol)
                        tried.append((name, 'symbol_kw'))
                        if resp and isinstance(resp, dict) and any(k.lower() in str(resp).lower() for k in ('ltp','open_interest','status','data')):
                            logger.debug("try_dhan_quote: got response from %s with symbol_kw", name)
                            return resp
                    except Exception:
                        pass
                # pattern 4: no args
                try:
                    resp = method()
                    tried.append((name, 'no_args'))
                    if resp and isinstance(resp, dict) and any(k.lower() in str(resp).lower() for k in ('ltp','open_interest','status','data')):
                        logger.debug("try_dhan_quote: got response from %s with no_args", name)
                        return resp
                except Exception:
                    pass
            except Exception as e:
                logger.debug("try_dhan_quote: attempted %s tries for %s failed: %s", tried, name, e)
                continue

        logger.debug("try_dhan_quote: no candidate method returned a recognizable result")
        return None
    except Exception as e:
        logger.exception("try_dhan_quote fatal: %s", e)
        return None

# ---------------- Use helper in spot price and options data -----------------
def get_spot_price_dhan(dhan, index_name):
    """Get spot price for NIFTY or BANKNIFTY using robust attempt to call SDK methods."""
    try:
        # Known mapping (may vary by account)
        security_id = 13 if index_name == 'NIFTY' else 25
        exchange_seg = getattr(dhan, 'NSE', None) or 'NSE'

        # Try our helper with security_id
        resp = try_dhan_quote(dhan, security_id=str(security_id), exchange_segment=exchange_seg, symbol=None)
        if not resp:
            # try querying by symbol name fallback (some SDKs expect symbol)
            symbol_name = 'NIFTY 50' if index_name == 'NIFTY' else 'BANKNIFTY'
            resp = try_dhan_quote(dhan, security_id=None, exchange_segment=exchange_seg, symbol=symbol_name)

        if not resp:
            logger.error("Failed to call any dhan quote method for spot price. See discover logs above.")
            return None

        # Normalize response: try to find LTP in resp or resp['data']
        data = resp.get('data') if isinstance(resp, dict) and 'data' in resp else resp
        ltp = None
        if isinstance(data, dict):
            ltp = data.get('LTP') or data.get('ltp') or data.get('last_price') or data.get('lastPrice') or None
        # if data is nested or list, try string search fallback
        if ltp is None:
            # crude attempt
            s = str(resp).lower()
            import re
            m = re.search(r'\"?ltp\"?\s*[:=]\s*\"?([\d\.]+)\"?', s)
            if m:
                ltp = m.group(1)

        if ltp is None:
            logger.error("Quote response didn't contain LTP. Response preview: %s", str(resp)[:400])
            return None

        ltp_val = float(ltp)
        logger.info("✅ %s Spot: ₹%s", index_name, f"{ltp_val:,.2f}")
        return ltp_val

    except Exception as e:
        logger.exception("❌ Error getting spot price: %s", e)
        return None

def get_option_chain_data(dhan, option_contracts):
    """Fetch option chain data with OI and Volume using try_dhan_quote helper."""
    try:
        if not option_contracts:
            return {}
        logger.info("📡 Fetching data for %d options...", len(option_contracts))
        result = {}
        # try to detect a good exchange segment constant
        exchange_seg = getattr(dhan, 'NSE_FNO', None) or getattr(dhan, 'FNO', None) or 'NSE_FNO'

        for opt in option_contracts[:40]:
            try:
                sec_id = str(opt.get('security_id') or opt.get('SEM_SMST_SECURITY_ID') or '')
                if not sec_id:
                    continue
                # primary: ask by security_id
                resp = try_dhan_quote(dhan, security_id=sec_id, exchange_segment=exchange_seg)
                if not resp:
                    # fallback: try using symbol
                    resp = try_dhan_quote(dhan, security_id=None, symbol=opt.get('symbol'), exchange_segment=exchange_seg)
                if not resp:
                    logger.warning("No quote for option sec_id=%s symbol=%s", sec_id, opt.get('symbol'))
                    continue

                # normalize
                data = resp.get('data') if isinstance(resp, dict) and 'data' in resp else resp
                ltp_val = (data.get('LTP') if isinstance(data, dict) else None) or \
                          (data.get('ltp') if isinstance(data, dict) else None) or \
                          (data.get('last_price') if isinstance(data, dict) else None) or 0
                oi_val = (data.get('open_interest') if isinstance(data, dict) else None) or \
                         (data.get('OI') if isinstance(data, dict) else None) or 0
                vol_val = (data.get('volume') if isinstance(data, dict) else None) or \
                          (data.get('Volume') if isinstance(data, dict) else None) or 0
                change_val = (data.get('change') if isinstance(data, dict) else None) or \
                             (data.get('Change') if isinstance(data, dict) else None) or 0

                result[sec_id] = {
                    'ltp': float(ltp_val) if ltp_val not in (None, '') else 0.0,
                    'oi': int(oi_val) if oi_val not in (None, '') else 0,
                    'volume': int(vol_val) if vol_val not in (None, '') else 0,
                    'change': float(change_val) if change_val not in (None, '') else 0.0
                }
                # small sleep to be gentle on API
                time.sleep(0.08)
            except Exception as e:
                logger.warning("Failed to get quote for %s: %s", opt.get('security_id'), e)
                continue

        logger.info("✅ Fetched data for %d options", len(result))
        return result

    except Exception as e:
        logger.exception("❌ Error fetching option data: %s", e)
        return {}

def calculate_strikes(spot_price, index_name, num_strikes=5):
    if index_name == 'NIFTY':
        strike_gap = 50
    else:
        strike_gap = 100
    atm = round(spot_price / strike_gap) * strike_gap
    return sorted([atm + (i * strike_gap) for i in range(-num_strikes, num_strikes+1)])

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
        # Log discovered methods for debugging
        items = discover_dhan_methods(dhan)
        logger.info("dhan object candidate methods: %s", items[:60])
        tele_send_http(TELE_CHAT_ID, f"✅ DhanHQ Option Chain Bot started!\nDetected methods: {items[:10]}")
    except Exception as e:
        logger.exception('❌ DhanHQ initialization failed: %s', e)
        tele_send_http(TELE_CHAT_ID, f'❌ DhanHQ init failed: {e}')
        return

    instruments = get_instruments(dhan)
    if not instruments:
        logger.error("❌ Failed to download instruments")
        tele_send_http(TELE_CHAT_ID, "❌ Failed to load instruments")
        return

    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info("\n" + "="*50)
            logger.info("🔄 Iteration #%d - %s", iteration, time.strftime('%Y-%m-%d %H:%M:%S'))
            logger.info("="*50)

            # NIFTY
            logger.info("--- Processing NIFTY ---")
            nifty_price = get_spot_price_dhan(dhan, 'NIFTY')
            if nifty_price and nifty_price > 0:
                nifty_expiry = (datetime.now() + timedelta(days=((3 - datetime.now().weekday()) % 7 or 7))).strftime('%Y-%m-%d')
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
                bn_expiry = (datetime.now() + timedelta(days=((2 - datetime.now().weekday()) % 7 or 7))).strftime('%Y-%m-%d')
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
