# main.py
import os
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify
import requests

# Import module only — don't import non-existent symbols at top-level.
# We'll inspect and call the right constructor inside make_dhan().
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
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL') or 60)  # seconds
STRIKE_COUNT = int(os.getenv('STRIKE_COUNT') or 10)   # total strikes around ATM (10 requested)
# Underlying ids (change if your account uses different IDs)
NIFTY_ID = int(os.getenv('NIFTY_ID') or 13)
BANKNIFTY_ID = int(os.getenv('BANKNIFTY_ID') or 25)
# Exchange segment key used by SDK/REST
IDX_KEY = os.getenv('IDX_KEY') or "IDX_I"

# Optional explicit expiries (YYYY-MM-DD). If empty, code will compute weekly expiry for NIFTY (Thu) and BANKNIFTY (Wed)
NIFTY_EXPIRY = os.getenv('NIFTY_EXPIRY') or ""
BANKNIFTY_EXPIRY = os.getenv('BANKNIFTY_EXPIRY') or ""

REQUIRED = [DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELE_TOKEN, TELE_CHAT_ID]

app = Flask(__name__)

def tele_send_http(chat_id: str, text: str):
    try:
        token = TELE_TOKEN
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN not set")
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
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
    """
    Robust client factory that handles multiple dhanhq package shapes.
    Tries common patterns:
      - dhanhq.dhanhq(client_id, access_token)
      - dhanhq.DhanContext(...) + dhanhq.dhanhq(ctx)
      - dhanhq.Client(...) variants
      - dhanhq.dhanhq({"client_id":..., "access_token":...})
    Raises RuntimeError with helpful info if it can't create a client.
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        raise RuntimeError("Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN")

    if dhanhq is None:
        raise RuntimeError("dhanhq package not importable in current environment")

    errors = []

    # 1) direct factory: dhanhq.dhanhq(client_id, access_token)
    try:
        if hasattr(dhanhq, "dhanhq") and callable(getattr(dhanhq, "dhanhq")):
            try:
                c = dhanhq.dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                logger.debug("Created dhan client via dhanhq.dhanhq(client_id, access_token)")
                return c
            except TypeError as te:
                errors.append(("dhanhq(... two args)", str(te)))
            except Exception as e:
                errors.append(("dhanhq(... two args) other", str(e)))
    except Exception as e:
        errors.append(("check dhanhq.dhanhq existence", str(e)))

    # 2) DhanContext style: dhanhq.DhanContext(...) then dhanhq.dhanhq(ctx)
    try:
        if hasattr(dhanhq, "DhanContext") and callable(getattr(dhanhq, "DhanContext")):
            try:
                ctx = dhanhq.DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                if hasattr(dhanhq, "dhanhq") and callable(getattr(dhanhq, "dhanhq")):
                    try:
                        c = dhanhq.dhanhq(ctx)
                        logger.debug("Created dhan client via DhanContext + dhanhq.dhanhq(ctx)")
                        return c
                    except Exception as e:
                        errors.append(("dhanhq(ctx) after DhanContext", str(e)))
                else:
                    return ctx
            except Exception as e:
                errors.append(("DhanContext(...) creation", str(e)))
    except Exception as e:
        errors.append(("check DhanContext existence", str(e)))

    # 3) Client class patterns
    try:
        if hasattr(dhanhq, "Client") and callable(getattr(dhanhq, "Client")):
            try:
                try:
                    c = dhanhq.Client(client_id=DHAN_CLIENT_ID, access_token=DHAN_ACCESS_TOKEN)
                    logger.debug("Created dhan client via dhanhq.Client(client_id=..., access_token=...)")
                    return c
                except TypeError:
                    c = dhanhq.Client(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                    logger.debug("Created dhan client via dhanhq.Client(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)")
                    return c
            except Exception as e:
                errors.append(("Client(...) creation", str(e)))
    except Exception as e:
        errors.append(("check Client existence", str(e)))

    # 4) try dict-style factory
    try:
        if hasattr(dhanhq, "dhanhq") and callable(getattr(dhanhq, "dhanhq")):
            try:
                c = dhanhq.dhanhq({"client_id": DHAN_CLIENT_ID, "access_token": DHAN_ACCESS_TOKEN})
                logger.debug("Created dhan client via dhanhq.dhanhq(dict)")
                return c
            except Exception as e:
                errors.append(("dhanhq(dict)", str(e)))
    except Exception as e:
        errors.append(("check dhanhq for dict-style", str(e)))

    # 5) last attempt: try any 'create' classmethod exposed
    try:
        for name in ("create","connect","from_credentials"):
            if hasattr(dhanhq, name) and callable(getattr(dhanhq, name)):
                try:
                    fn = getattr(dhanhq, name)
                    c = fn(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
                    logger.debug("Created dhan client via dhanhq.%s()", name)
                    return c
                except Exception as e:
                    errors.append((f"{name}(id,token)", str(e)))
    except Exception as e:
        errors.append(("create-like attempts", str(e)))

    # If none succeeded, raise informative error
    available = ", ".join(sorted(n for n in dir(dhanhq) if not n.startswith("_")))
    err_text = f"Could not construct dhanhq client. Available attrs: {available}. Attempts/errors: {errors}"
    raise RuntimeError(err_text)

# ---------------- expiry helpers ----------------
def weekly_expiry_for_index(index_name):
    # NIFTY weekly expiry = Thursday (weekday 3), BANKNIFTY weekly expiry = Wednesday (weekday 2)
    today = datetime.now().date()
    if index_name == "NIFTY":
        target = 3
    else:
        target = 2
    days_ahead = (target - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    exp = today + timedelta(days=days_ahead)
    return exp.strftime("%Y-%m-%d")

# ---------------- robust calls to SDK and REST fallback ----------------
def try_quote_data(dhan, exchange_key, underlying_id):
    """
    Try calling dhan.quote_data correctly.
    Preferred payload: {'securities': {'IDX_I': [13]}}
    Returns parsed response or None.
    """
    payload = {"securities": {exchange_key: [int(underlying_id)]}}
    try:
        resp = None
        try:
            resp = dhan.quote_data(securities=payload["securities"])
        except TypeError:
            resp = dhan.quote_data(payload)
        except Exception as e:
            logger.debug("quote_data first attempts error: %s", e)
            try:
                resp = dhan.quote_data(payload["securities"])
            except Exception as e2:
                logger.debug("quote_data fallback error: %s", e2)
                resp = None
        return resp
    except Exception as e:
        logger.exception("try_quote_data fatal: %s", e)
        return None

def rest_option_chain(underlying_id, exchange_key, expiry):
    """Fallback REST call to Dhan option-chain endpoint (uses client-id + access-token headers)."""
    try:
        url = "https://api.dhan.co/v2/option-chain"
        headers = {
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID,
            "Content-Type": "application/json"
        }
        payload = {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": exchange_key, "Expiry": expiry}
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            logger.warning("REST option-chain returned %s: %s", r.status_code, r.text[:300])
            return None
    except Exception as e:
        logger.exception("REST option chain call failed: %s", e)
        return None

def try_option_chain_sdk(dhan, underlying_id, expiry, exchange_key):
    """
    Try various argument styles for dhan.option_chain.
    Return response dict or None.
    """
    tries = []
    candidates = [
        {"under_security_id": int(underlying_id), "under_exchange_segment": exchange_key, "expiry": expiry},
        {"underlying_security_id": int(underlying_id), "under_exchange_segment": exchange_key, "expiry": expiry},
        {"under_security_id": int(underlying_id), "under_exchange_segment": exchange_key, "Expiry": expiry},
        {"UNDERLYING": int(underlying_id), "Expiry": expiry},
        {"instrument": int(underlying_id), "segment": exchange_key, "expiry": expiry},
    ]
    for args in candidates:
        try:
            try:
                resp = dhan.option_chain(**args)
                logger.debug("option_chain sdk returned with args %s", list(args.keys()))
                return resp
            except TypeError:
                resp = dhan.option_chain(args)
                logger.debug("option_chain sdk (dict arg) returned with args %s", list(args.keys()))
                return resp
            except Exception as e:
                tries.append((args, str(e)))
                logger.debug("option_chain try failed for %s : %s", args, e)
                continue
        except Exception as e:
            logger.debug("option_chain outer fail: %s", e)
            continue
    logger.debug("option_chain sdk tries: %s", tries)
    return None

# ---------------- normalization & spot extraction helpers ----------------
def safe_get_ltp_from_quote_response(resp):
    try:
        if not resp:
            return None
        # resp may be dict with 'data'
        if isinstance(resp, dict):
            data = resp.get("data", resp.get("Data", resp))
            # if data is dict containing exchange key
            if isinstance(data, dict):
                # try to find first numeric last price
                for v in data.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        # find LTP in v[0]
                        item = v[0]
                        for k in ("LTP","ltp","lastPrice","last_price","last"):
                            if k in item and item[k] not in (None, ""):
                                try:
                                    return float(item[k])
                                except:
                                    pass
                # direct fields
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
        # fallback search numbers
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
    """
    Robust extraction of underlying spot price from many possible shapes.
    - Recursively searches dict/list for keys that hint at underlying/spot/ltp/last.
    - If finds numeric-like string/value, returns that.
    - If none found, scans for strikes and returns median strike (inferred).
    Returns (spot: float or None, preview_str: str, inferred: bool)
    """
    try:
        preview = ""
        if not raw:
            return None, preview, False
        # small preview for logs
        try:
            preview = json.dumps(raw)[:1200]
        except Exception:
            preview = str(raw)[:1200]

        # helper: try convert to float
        def to_num(v):
            try:
                if v is None: return None
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).strip().replace(',', '')
                if s == "":
                    return None
                # if purely non-numeric, ignore
                # allow negative and decimal
                # remove plus signs
                if any(c.isalpha() for c in s) and not any(ch.isdigit() for ch in s):
                    return None
                return float(s)
            except:
                return None

        # keys that strongly indicate underlying price
        key_patterns = ("underlying","underlyinglast","underlyinglastprice",
                        "underlying_price","underlyingltp","spot","ltp","lastprice","last_price","last")

        # recursive search
        found_values = []
        def recurse(obj, path=""):
            if obj is None:
                return
            if isinstance(obj, dict):
                for k,v in obj.items():
                    key_low = str(k).lower()
                    num = None
                    for pat in key_patterns:
                        if pat in key_low:
                            num = to_num(v)
                            if num is not None:
                                found_values.append((num, path + "/" + str(k)))
                                return  # prefer first strong hit
                    recurse(v, path + "/" + str(k))
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:200]):  # limit length
                    recurse(item, f"{path}[{i}]")
            else:
                num = to_num(obj)
                if num is not None:
                    found_values.append((num, path))

        recurse(raw)

        if found_values:
            num, where = found_values[0]
            return float(num), preview, False

        # if nothing direct found, try to infer from strikes present in option rows
        strikes = []
        data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
        target = data
        if isinstance(data, dict):
            for k in ("options","optionchain","records","rows","data","optionslist"):
                if k in data and isinstance(data[k], list):
                    target = data[k]
                    break
        if isinstance(target, list):
            for item in target:
                if not isinstance(item, dict):
                    continue
                for k in ("strike","strikeprice","strike_price","sem_strike_price","StrikePrice"):
                    if k in item:
                        try:
                            s = int(float(item[k])); strikes.append(s)
                        except:
                            pass
                # try inside CE/PE blocks
                for side in ("CE","PE","call","put"):
                    if side in item and isinstance(item[side], dict):
                        for kk in ("strike","StrikePrice"):
                            if kk in item[side]:
                                try:
                                    s = int(float(item[side][kk])); strikes.append(s)
                                except:
                                    pass
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
    # if total is even, we'll take half below and half above (e.g. 10 -> -5..+4)
    offsets = list(range(-half, half))
    strikes = sorted([int(atm + i*gap) for i in offsets])
    return strikes

def parse_option_chain_raw(raw):
    """
    Convert raw option-chain response to a list/dict of rows with keys:
    strike (int), CE, PE with ltp/oi/volume
    """
    rows = []
    if not raw:
        return rows
    data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
    # Common shapes: data might include 'options' list or list of rows
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
                l = 0.0
                oi = 0
                vol = 0
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
                            try: oi=int(float(o[kk])); break
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

# ---------------- formatting message ----------------
def format_for_telegram(index_name, spot, expiry, strikes, strike_rows, spot_inferred=False):
    # strike_rows: list of {'strike':int,'CE':{..},'PE':{..}}
    m = {r['strike']: {'CE': r.get('CE',{}), 'PE': r.get('PE',{})} for r in strike_rows}
    lines = []
    inferred_note = " (inferred)" if spot_inferred else ""
    lines.append(f"📊 <b>{index_name}</b>  | Spot: <b>₹{spot:,.2f}</b>{inferred_note}  | Expiry: {expiry}")
    lines.append("<code>  CE_LTP   CE_OI    STRIKE    PE_LTP   PE_OI</code>")
    lines.append("─"*60)
    for s in sorted(strikes):
        d = m.get(s, {'CE':{'ltp':0,'oi':0}, 'PE':{'ltp':0,'oi':0}})
        ce = d['CE']; pe = d['PE']
        try:
            ce_ltp = float(ce.get('ltp', 0.0))
        except:
            ce_ltp = 0.0
        try:
            pe_ltp = float(pe.get('ltp', 0.0))
        except:
            pe_ltp = 0.0
        try:
            ce_oi = int(ce.get('oi', 0))
        except:
            ce_oi = 0
        try:
            pe_oi = int(pe.get('oi', 0))
        except:
            pe_oi = 0
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
        try:
            logger.debug("dhan methods: %s", [m for m in dir(dhan) if callable(getattr(dhan,m))][:80])
        except Exception:
            pass
        tele_send_http(TELE_CHAT_ID, "✅ DhanHQ Option Chain Bot started!")
    except Exception as e:
        logger.exception("Dhan init failed: %s", e)
        tele_send_http(TELE_CHAT_ID, f"❌ Dhan init failed: {e}")
        return

    iteration = 0
    while True:
        iteration += 1
        logger.info("=== Iteration #%d ===", iteration)
        try:
            # prepare expiries
            nifty_expiry = NIFTY_EXPIRY or weekly_expiry_for_index("NIFTY")
            bank_expiry = BANKNIFTY_EXPIRY or weekly_expiry_for_index("BANKNIFTY")

            # --- NIFTY ---
            spot_n = None
            spot_n_inferred = False
            try:
                resp_q = try_quote_data(dhan, IDX_KEY, NIFTY_ID)
                spot_n = safe_get_ltp_from_quote_response(resp_q)
                logger.debug("NIFTY quote_data resp preview: %s", str(resp_q)[:300])
            except Exception as e:
                logger.debug("quote_data exception for NIFTY: %s", e)

            raw_chain_n = None
            if spot_n is None:
                raw_chain_n = try_option_chain_sdk(dhan, NIFTY_ID, nifty_expiry, IDX_KEY)
                if raw_chain_n:
                    s, preview, inferred = extract_spot_from_chain(raw_chain_n)
                    logger.info("NIFTY raw_chain preview (first 1200 chars): %s", preview)
                    if s is not None:
                        spot_n = s
                        spot_n_inferred = bool(inferred)

                # if still none, fallback REST option-chain (which in many examples returns option + underlying)
                if spot_n is None and raw_chain_n is None:
                    raw_chain_n = rest_option_chain(NIFTY_ID, IDX_KEY, nifty_expiry)
                    if raw_chain_n:
                        s, preview, inferred = extract_spot_from_chain(raw_chain_n)
                        logger.info("NIFTY REST raw_chain preview (first 1200 chars): %s", preview)
                        if s is not None:
                            spot_n = s
                            spot_n_inferred = bool(inferred)

            # if still no raw_chain_n, try SDK again
            if raw_chain_n is None:
                raw_chain_n = try_option_chain_sdk(dhan, NIFTY_ID, nifty_expiry, IDX_KEY) or rest_option_chain(NIFTY_ID, IDX_KEY, nifty_expiry)

            # if we have spot and chain, parse & send
            if raw_chain_n and spot_n is not None:
                # choose ATM and strikes
                gap = 50
                atm = round(spot_n / gap) * gap
                strikes = choose_strikes_around_atm(atm, gap, total=STRIKE_COUNT)
                parsed_rows = parse_option_chain_raw(raw_chain_n)
                # filter parsed_rows for required strikes
                filtered = [r for r in parsed_rows if r['strike'] in strikes]
                # fill missing strikes with empty template
                for s in strikes:
                    if not any(r['strike']==s for r in filtered):
                        filtered.append({'strike':s,'CE':{'ltp':0.0,'oi':0,'volume':0},'PE':{'ltp':0.0,'oi':0,'volume':0}})
                text = format_for_telegram("NIFTY 50", spot_n, nifty_expiry, strikes, filtered, spot_inferred=spot_n_inferred)
                tele_send_http(TELE_CHAT_ID, text)
                logger.info("Sent NIFTY option chain (strikes=%d) to Telegram", len(strikes))
            else:
                logger.warning("Could not fetch complete NIFTY data (spot:%s, chain:%s)", bool(spot_n), bool(raw_chain_n))
                tele_send_http(TELE_CHAT_ID, f"⚠️ NIFTY fetch incomplete. spot={spot_n is not None}, chain={raw_chain_n is not None}")

            time.sleep(1)  # small gap

            # --- BANKNIFTY ---
            spot_b = None
            spot_b_inferred = False
            try:
                resp_qb = try_quote_data(dhan, IDX_KEY, BANKNIFTY_ID)
                spot_b = safe_get_ltp_from_quote_response(resp_qb)
                logger.debug("BANKNIFTY quote_data resp preview: %s", str(resp_qb)[:300])
            except Exception as e:
                logger.debug("quote_data exception for BANKNIFTY: %s", e)

            raw_chain_b = None
            if spot_b is None:
                raw_chain_b = try_option_chain_sdk(dhan, BANKNIFTY_ID, bank_expiry, IDX_KEY)
                if raw_chain_b:
                    s, preview, inferred = extract_spot_from_chain(raw_chain_b)
                    logger.info("BANKNIFTY raw_chain preview (first 1200 chars): %s", preview)
                    if s is not None:
                        spot_b = s
                        spot_b_inferred = bool(inferred)

                if spot_b is None and raw_chain_b is None:
                    raw_chain_b = rest_option_chain(BANKNIFTY_ID, IDX_KEY, bank_expiry)
                    if raw_chain_b:
                        s, preview, inferred = extract_spot_from_chain(raw_chain_b)
                        logger.info("BANKNIFTY REST raw_chain preview (first 1200 chars): %s", preview)
                        if s is not None:
                            spot_b = s
                            spot_b_inferred = bool(inferred)

            if raw_chain_b is None:
                raw_chain_b = try_option_chain_sdk(dhan, BANKNIFTY_ID, bank_expiry, IDX_KEY) or rest_option_chain(BANKNIFTY_ID, IDX_KEY, bank_expiry)

            if raw_chain_b and spot_b is not None:
                gapb = 100
                atmb = round(spot_b / gapb) * gapb
                strikes_b = choose_strikes_around_atm(atmb, gapb, total=STRIKE_COUNT)
                parsed_b = parse_option_chain_raw(raw_chain_b)
                filtered_b = [r for r in parsed_b if r['strike'] in strikes_b]
                for s in strikes_b:
                    if not any(r['strike']==s for r in filtered_b):
                        filtered_b.append({'strike':s,'CE':{'ltp':0.0,'oi':0,'volume':0},'PE':{'ltp':0.0,'oi':0,'volume':0}})
                textb = format_for_telegram("BANK NIFTY", spot_b, bank_expiry, strikes_b, filtered_b, spot_inferred=spot_b_inferred)
                tele_send_http(TELE_CHAT_ID, textb)
                logger.info("Sent BANKNIFTY option chain (strikes=%d) to Telegram", len(strikes_b))
            else:
                logger.warning("Could not fetch complete BANKNIFTY data (spot:%s, chain:%s)", bool(spot_b), bool(raw_chain_b))
                tele_send_http(TELE_CHAT_ID, f"⚠️ BANKNIFTY fetch incomplete. spot={spot_b is not None}, chain={raw_chain_b is not None}")

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
        'service': 'DhanHQ Option Chain Bot',
        'thread_alive': thread.is_alive(),
        'poll_interval': POLL_INTERVAL,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health():
    return jsonify({'status':'ok', 'thread_alive': thread.is_alive()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
