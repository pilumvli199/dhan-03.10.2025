# main.py (SDK-only, minimal & robust)
import os
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify

# Try import dependencies
try:
    import requests
except Exception:
    requests = None

try:
    import dhanhq
except Exception:
    dhanhq = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('dhanhq-option-chain-bot')

# === Config / Env ===
DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')
TELE_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELE_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

POLL_INTERVAL = int(os.getenv('POLL_INTERVAL') or 60)
STRIKE_COUNT = int(os.getenv('STRIKE_COUNT') or 10)
NIFTY_ID = int(os.getenv('NIFTY_ID') or 13)
BANKNIFTY_ID = int(os.getenv('BANKNIFTY_ID') or 25)
IDX_KEY = os.getenv('IDX_KEY') or "IDX_I"

NIFTY_EXPIRY = os.getenv('NIFTY_EXPIRY') or ""
BANKNIFTY_EXPIRY = os.getenv('BANKNIFTY_EXPIRY') or ""

INCOMPLETE_ALERT_COOLDOWN = int(os.getenv('INCOMPLETE_ALERT_COOLDOWN') or 600)  # seconds

REQUIRED = [DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELE_TOKEN, TELE_CHAT_ID]

app = Flask(__name__)

def tele_send_http(chat_id: str, text: str):
    if not TELE_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return False
    if requests is None:
        logger.error("requests lib not available")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.warning("Telegram API returned %s: %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.exception("Failed to send Telegram message: %s", e)
        return False

# ---------------- Dhan client creation ----------------
def make_dhan():
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        raise RuntimeError("Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN")
    if dhanhq is None:
        raise RuntimeError("dhanhq package not importable in current environment")
    # try common constructors
    try:
        if hasattr(dhanhq, "dhanhq") and callable(getattr(dhanhq, "dhanhq")):
            try:
                return dhanhq.dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
            except:
                pass
        if hasattr(dhanhq, "DhanContext") and callable(getattr(dhanhq, "DhanContext")):
            try:
                ctx = dhanhq.DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                if hasattr(dhanhq, "dhanhq") and callable(getattr(dhanhq, "dhanhq")):
                    return dhanhq.dhanhq(ctx)
                return ctx
            except:
                pass
        if hasattr(dhanhq, "Client") and callable(getattr(dhanhq, "Client")):
            try:
                return dhanhq.Client(client_id=DHAN_CLIENT_ID, access_token=DHAN_ACCESS_TOKEN)
            except:
                try:
                    return dhanhq.Client(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                except:
                    pass
    except Exception as e:
        logger.debug("make_dhan attempts failed: %s", e)
    raise RuntimeError("Could not construct dhanhq client with available module shape.")

# ---------------- expiry helpers ----------------
def weekly_expiry_for_index(index_name):
    today = datetime.now().date()
    target = 3 if index_name == "NIFTY" else 2
    days_ahead = (target - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)

def expiry_variants(date_obj):
    return [date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%d-%b-%Y"), date_obj.strftime("%Y%m%d"), date_obj.strftime("%d-%m-%Y")]

# ---------------- SDK helpers ----------------
def try_quote_data(dhan, exchange_key, underlying_id):
    payload = {"securities": {exchange_key: [int(underlying_id)]}}
    try:
        try:
            return dhan.quote_data(securities=payload["securities"])
        except TypeError:
            return dhan.quote_data(payload)
        except Exception:
            try:
                return dhan.quote_data(payload["securities"])
            except:
                return None
    except Exception as e:
        logger.debug("try_quote_data fatal: %s", e)
        return None

def try_option_chain_sdk(dhan, underlying_id, expiry_date, exchange_key):
    expiry_vals = expiry_variants(expiry_date)
    bases = [
        {"under_security_id": int(underlying_id), "under_exchange_segment": exchange_key},
        {"underlying_security_id": int(underlying_id), "under_exchange_segment": exchange_key},
        {"UNDERLYING": int(underlying_id)},
        {"instrument": int(underlying_id), "segment": exchange_key},
    ]
    attempts = []
    for base in bases:
        for exp in expiry_vals:
            for expk in ("expiry","Expiry","ExpiryDate","Expiry_Date"):
                args = dict(base); args[expk] = exp
                try:
                    try:
                        resp = dhan.option_chain(**args)
                        if isinstance(resp, dict) and resp.get("status") and str(resp.get("status")).lower() == "failure":
                            attempts.append((args, "failure"))
                            continue
                        return resp
                    except TypeError:
                        resp = dhan.option_chain(args)
                        if isinstance(resp, dict) and resp.get("status") and str(resp.get("status")).lower() == "failure":
                            attempts.append((args, "failure"))
                            continue
                        return resp
                    except Exception as e:
                        attempts.append((args, str(e)))
                        continue
                except Exception as e:
                    attempts.append((args, str(e)))
                    continue
    logger.debug("option_chain sdk tries: %s", attempts)
    return None

# ---------------- parsing / formatting ----------------
def safe_get_ltp_from_quote_response(resp):
    try:
        if not resp:
            return None
        if isinstance(resp, dict):
            data = resp.get("data", resp.get("Data", resp))
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        item = v[0]
                        for k in ("LTP","ltp","lastPrice","last_price","last"):
                            if k in item and item[k] not in (None, ""):
                                try:
                                    return float(item[k])
                                except:
                                    pass
                for k in ("LTP","ltp","lastPrice","last_price","last"):
                    if k in data and data[k] not in (None, ""):
                        try:
                            return float(data[k])
                        except:
                            pass
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                item = data[0]
                for k in ("LTP","ltp","lastPrice","last_price","last"):
                    if k in item and item[k] not in (None, ""):
                        try:
                            return float(item[k])
                        except:
                            pass
        s = str(resp)
        import re
        m = re.search(r'ltp[^\d]*([\d]+\.[\d]+|[\d]+)', s, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None
    except Exception as e:
        logger.debug("safe_get_ltp fail: %s", e)
        return None

def extract_spot_from_chain(raw):
    try:
        preview = ""
        if not raw:
            return None, preview, False
        try:
            preview = json.dumps(raw)[:1200]
        except:
            preview = str(raw)[:1200]
        if isinstance(raw, dict) and raw.get("status") and str(raw.get("status")).lower() == "failure":
            return None, preview, False
        def to_num(v):
            try:
                if v is None: return None
                if isinstance(v, (int,float)): return float(v)
                s = str(v).strip().replace(',', '')
                if s == "": return None
                if any(c.isalpha() for c in s) and not any(ch.isdigit() for ch in s):
                    return None
                return float(s)
            except:
                return None
        patterns = ("underlying","underlyinglast","underlyinglastprice","underlying_price","underlyingltp","spot","ltp","lastprice","last_price","last")
        found=[]
        def recurse(o):
            if o is None: return
            if isinstance(o, dict):
                for k,v in o.items():
                    lk = str(k).lower()
                    for p in patterns:
                        if p in lk:
                            n = to_num(v)
                            if n is not None:
                                found.append(n); return
                    recurse(v)
            elif isinstance(o, list):
                for it in o[:200]:
                    recurse(it)
            else:
                n = to_num(o)
                if n is not None:
                    found.append(n)
        recurse(raw)
        if found:
            return float(found[0]), preview, False
        strikes=[]
        data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
        target = data
        if isinstance(data, dict):
            for k in ("options","optionchain","records","rows","data","optionslist"):
                if k in data and isinstance(data[k], list):
                    target = data[k]; break
        if isinstance(target, list):
            for item in target:
                if not isinstance(item, dict): continue
                for k in ("strike","strikeprice","strike_price","sem_strike_price","StrikePrice"):
                    if k in item:
                        try: strikes.append(int(float(item[k])))
                        except: pass
                for side in ("CE","PE","call","put"):
                    if side in item and isinstance(item[side], dict):
                        for kk in ("strike","StrikePrice"):
                            if kk in item[side]:
                                try: strikes.append(int(float(item[side][kk])))
                                except: pass
        if strikes:
            strikes_sorted = sorted(set(strikes))
            mid = strikes_sorted[len(strikes_sorted)//2]
            return float(mid), preview, True
        return None, preview, False
    except Exception as e:
        logger.exception("extract_spot_from_chain fail: %s", e)
        return None, "", False

def choose_strikes_around_atm(atm, gap, total=10):
    half = total // 2
    offsets = list(range(-half, half))
    strikes = sorted([int(atm + i*gap) for i in offsets])
    return strikes

def parse_option_chain_raw(raw):
    rows = []
    if not raw:
        return rows
    data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
    if isinstance(data, dict):
        for k in ("options","optionChain","records","rows","data","optionsList"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            strike = None
            for k in ("strike","StrikePrice","strikePrice","SEM_STRIKE_PRICE"):
                if k in item:
                    try:
                        strike = int(float(item[k])); break
                    except:
                        pass
            if strike is None:
                if 'CE' in item and isinstance(item['CE'], dict):
                    try:
                        strike = int(float(item['CE'].get('strike') or item['CE'].get('StrikePrice') or item['CE'].get('SEM_STRIKE_PRICE')))
                    except:
                        strike = None
            if strike is None:
                continue
            ce = item.get('CE') or item.get('call') or None
            pe = item.get('PE') or item.get('put') or None
            def ext(o):
                if not o or not isinstance(o, dict):
                    return {'ltp':0.0,'oi':0,'volume':0}
                l = 0.0; oi = 0; vol = 0
                for k in ('LTP','ltp','lastPrice','last_price'):
                    if k in o and o[k] not in (None,''):
                        try:
                            l = float(o[k]); break
                        except: pass
                for k in ('openInterest','OI','open_interest'):
                    if k in o and o[k] not in (None,''):
                        try:
                            oi = int(float(o[k])); break
                        except: pass
                for k in ('volume','Volume'):
                    if k in o and o[k] not in (None,''):
                        try:
                            vol = int(float(o[k])); break
                        except: pass
                return {'ltp':l,'oi':oi,'volume':vol}
            rows.append({'strike':strike,'CE':ext(ce),'PE':ext(pe)})
    elif isinstance(data, dict):
        for k,v in data.items():
            try:
                s = int(float(k))
                ce = v.get('CE') if isinstance(v, dict) else None
                pe = v.get('PE') if isinstance(v, dict) else None
                def ext(o):
                    if not o: return {'ltp':0.0,'oi':0,'volume':0}
                    l=0.0; oi=0; vol=0
                    for kk in ('LTP','ltp','lastPrice'):
                        if kk in o and o[kk] not in (None,''):
                            try: l=float(o[kk]); break
                            except: pass
                    for kk in ('openInterest','OI'):
                        if kk in o and o[kk] not in (None,''):
                            try: oi=int(float(o[kk]); break
                            except: pass
                    for kk in ('volume','Volume'):
                        if kk in o and o[kk] not in (None,''):
                            try: vol=int(float(o[kk])); break
                            except: pass
                    return {'ltp':l,'oi':oi,'volume':vol}
                rows.append({'strike':s,'CE':ext(ce),'PE':ext(pe)})
            except:
                continue
    return rows

def format_for_telegram(index_name, spot, expiry, strikes, strike_rows, spot_inferred=False):
    m = {r['strike']: {'CE': r.get('CE',{}), 'PE': r.get('PE',{})} for r in strike_rows}
    lines = []
    inferred_note = " (inferred)" if spot_inferred else ""
    lines.append(f"📊 <b>{index_name}</b>  | Spot: <b>₹{spot:,.2f}</b>{inferred_note}  | Expiry: {expiry}")
    lines.append("<code>  CE_LTP   CE_OI    STRIKE    PE_LTP   PE_OI</code>")
    lines.append("─"*60)
    for s in sorted(strikes):
        d = m.get(s, {'CE':{'ltp':0,'oi':0}, 'PE':{'ltp':0,'oi':0}})
        ce = d['CE']; pe = d['PE']
        try: ce_ltp = float(ce.get('ltp', 0.0))
        except: ce_ltp = 0.0
        try: pe_ltp = float(pe.get('ltp', 0.0))
        except: pe_ltp = 0.0
        try: ce_oi = int(ce.get('oi', 0))
        except: ce_oi = 0
        try: pe_oi = int(pe.get('oi', 0))
        except: pe_oi = 0
        lines.append(f"<code>{ce_ltp:8.1f} {ce_oi:8d} {int(s):10d} {pe_ltp:8.1f} {pe_oi:8d}</code>")
    lines.append("─"*60)
    lines.append(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)

# ---------------- main bot loop ----------------
def bot_loop():
    if not all(REQUIRED):
        logger.error("Missing required env vars: DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    try:
        dhan = make_dhan()
        logger.info("✅ Dhan client initialized")
        logger.info("Dhan client type: %s", type(dhan))
        tele_send_http(TELE_CHAT_ID, "✅ DhanHQ Option Chain Bot started (SDK-only)!")
    except Exception as e:
        logger.exception("Dhan init failed: %s", e)
        tele_send_http(TELE_CHAT_ID, f"❌ Dhan init failed: {e}")
        return

    iteration = 0
    last_incomplete_alert = 0.0
    while True:
        iteration += 1
        logger.info("=== Iteration #%d ===", iteration)
        try:
            nifty_expiry_date = datetime.strptime(NIFTY_EXPIRY, "%Y-%m-%d").date() if NIFTY_EXPIRY else weekly_expiry_for_index("NIFTY")
            bank_expiry_date = datetime.strptime(BANKNIFTY_EXPIRY, "%Y-%m-%d").date() if BANKNIFTY_EXPIRY else weekly_expiry_for_index("BANKNIFTY")

            # NIFTY
            spot_n = None
            spot_n_inferred = False
            try:
                resp_q = try_quote_data(dhan, IDX_KEY, NIFTY_ID)
                spot_n = safe_get_ltp_from_quote_response(resp_q)
            except Exception as e:
                logger.debug("quote_data exception for NIFTY: %s", e)

            raw_chain_n = try_option_chain_sdk(dhan, NIFTY_ID, nifty_expiry_date, IDX_KEY)
            if raw_chain_n:
                s, preview, inferred = extract_spot_from_chain(raw_chain_n)
                logger.info("NIFTY raw_chain preview (first 1200 chars): %s", preview)
                if s is not None:
                    spot_n = s
                    spot_n_inferred = bool(inferred)

            if raw_chain_n and spot_n is not None:
                gap = 50
                atm = round(spot_n / gap) * gap
                strikes = choose_strikes_around_atm(atm, gap, total=STRIKE_COUNT)
                parsed_rows = parse_option_chain_raw(raw_chain_n)
                filtered = [r for r in parsed_rows if r['strike'] in strikes]
                for s in strikes:
                    if not any(r['strike']==s for r in filtered):
                        filtered.append({'strike':s,'CE':{'ltp':0.0,'oi':0,'volume':0},'PE':{'ltp':0.0,'oi':0,'volume':0}})
                text = format_for_telegram("NIFTY 50", spot_n, nifty_expiry_date.strftime("%Y-%m-%d"), strikes, filtered, spot_inferred=spot_n_inferred)
                tele_send_http(TELE_CHAT_ID, text)
                logger.info("Sent NIFTY option chain (strikes=%d) to Telegram", len(strikes))
            else:
                logger.warning("Could not fetch complete NIFTY data (spot:%s, chain:%s)", bool(spot_n), bool(raw_chain_n))
                if time.time() - last_incomplete_alert > INCOMPLETE_ALERT_COOLDOWN:
                    tele_send_http(TELE_CHAT_ID, f"⚠️ NIFTY fetch incomplete. spot={spot_n is not None}, chain={raw_chain_n is not None}")
                    last_incomplete_alert = time.time()

            time.sleep(1)

            # BANKNIFTY
            spot_b = None
            spot_b_inferred = False
            try:
                resp_qb = try_quote_data(dhan, IDX_KEY, BANKNIFTY_ID)
                spot_b = safe_get_ltp_from_quote_response(resp_qb)
            except Exception as e:
                logger.debug("quote_data exception for BANKNIFTY: %s", e)

            raw_chain_b = try_option_chain_sdk(dhan, BANKNIFTY_ID, bank_expiry_date, IDX_KEY)
            if raw_chain_b:
                s, preview, inferred = extract_spot_from_chain(raw_chain_b)
                logger.info("BANKNIFTY raw_chain preview (first 1200 chars): %s", preview)
                if s is not None:
                    spot_b = s
                    spot_b_inferred = bool(inferred)

            if raw_chain_b and spot_b is not None:
                gapb = 100
                atmb = round(spot_b / gapb) * gapb
                strikes_b = choose_strikes_around_atm(atmb, gapb, total=STRIKE_COUNT)
                parsed_b = parse_option_chain_raw(raw_chain_b)
                filtered_b = [r for r in parsed_b if r['strike'] in strikes_b]
                for s in strikes_b:
                    if not any(r['strike']==s for r in filtered_b):
                        filtered_b.append({'strike':s,'CE':{'ltp':0.0,'oi':0,'volume':0},'PE':{'ltp':0.0,'oi':0,'volume':0}})
                textb = format_for_telegram("BANK NIFTY", spot_b, bank_expiry_date.strftime("%Y-%m-%d"), strikes_b, filtered_b, spot_inferred=spot_b_inferred)
                tele_send_http(TELE_CHAT_ID, textb)
                logger.info("Sent BANKNIFTY option chain (strikes=%d) to Telegram", len(strikes_b))
            else:
                logger.warning("Could not fetch complete BANKNIFTY data (spot:%s, chain:%s)", bool(spot_b), bool(raw_chain_b))
                if time.time() - last_incomplete_alert > INCOMPLETE_ALERT_COOLDOWN:
                    tele_send_http(TELE_CHAT_ID, f"⚠️ BANKNIFTY fetch incomplete. spot={spot_b is not None}, chain={raw_chain_b is not None}")
                    last_incomplete_alert = time.time()

        except Exception as e:
            logger.exception("Error in bot loop: %s", e)
            tele_send_http(TELE_CHAT_ID, f"⚠️ Bot error: {str(e)[:200]}")

        time.sleep(POLL_INTERVAL)

# start background thread
thread = threading.Thread(target=bot_loop, daemon=True)
thread.start()

@app.route('/')
def index():
    return jsonify({
        'service': 'DhanHQ Option Chain Bot (SDK-only)',
        'thread_alive': thread.is_alive(),
        'poll_interval': POLL_INTERVAL,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health():
    return jsonify({'status':'ok', 'thread_alive': thread.is_alive()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
