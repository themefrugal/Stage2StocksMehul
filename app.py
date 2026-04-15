#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   Stage 2 Breakout Screener — Nifty Total Market (750)     ║
║   7-Point Weinstein Scoring | Full-Universe Daily Cache    ║
║   DATA: constituents.json | HOLIDAYS: nse_holidays.json    ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")
HISTORY_PERIOD = "2y"
MIN_VOLUME = 100_000
HH_HL_LOOKBACK = 50          # Change here if needed
MA_RISING_LOOKBACK = 10      # Change here if needed
RESULT_CACHE_DIR = "daily_cache"
os.makedirs(RESULT_CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# HOLIDAY & WEEKEND CHECKER (Info only, non-blocking)
# ──────────────────────────────────────────────
def load_nse_holidays() -> set:
    path = os.path.join(os.path.dirname(__file__), "nse_holidays.json")
    if not os.path.exists(path): return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    holidays = set()
    for segment in data.values():
        for entry in segment:
            date_str = entry.get("tradingDate ", entry.get("tradingDate", "")).strip()
            try:
                dt = datetime.strptime(date_str, "%d-%b-%Y")
                holidays.add(dt.strftime("%Y-%m-%d"))
            except ValueError:
                continue
    return holidays

def is_weekend_or_holiday(date_str: str, holidays: set) -> bool:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5 or date_str in holidays
    except ValueError:
        return False

# ──────────────────────────────────────────────
# PERSISTENT CACHE UTILS
# ──────────────────────────────────────────────
def get_target_date_key() -> str:
    """<7 PM IST → yesterday, >=7 PM IST → today."""
    now = datetime.now(IST)
    if now.hour >= 19:
        return now.strftime("%Y-%m-%d")
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")

def load_json_cache(key: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path):
            df = pd.read_json(path, orient="records")
            return df if not df.empty else None
    except Exception:
        return None
    return None

def save_json_cache(df: pd.DataFrame, key: str):
    path = os.path.join(RESULT_CACHE_DIR, f"{key}.json")
    df.to_json(path, orient="records")

def load_latest_fallback(target_key: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        files = sorted([f.replace(".json", "") for f in os.listdir(RESULT_CACHE_DIR) if f.endswith(".json")], reverse=True)
        for f in files:
            if f <= target_key:
                df = load_json_cache(f)
                if df is not None: return df, f
    except Exception:
        pass
    return None, None

def _is_yfinance_fresh(raw_df: pd.DataFrame, target_date_str: str) -> bool:
    """Validates if yfinance returned complete EOD data for the target date."""
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            first_ticker = raw_df.columns.levels[0][0]
            sub = raw_df[first_ticker].dropna(how='all')
        else:
            sub = raw_df.dropna(how='all')
            
        if sub.empty: return False
        latest_date = sub.index[-1].date().strftime("%Y-%m-%d")
        if latest_date != target_date_str: return False
        vol = sub["Volume"].iloc[-1]
        return not (pd.isna(vol) or vol <= 0)
    except Exception:
        return False

def fetch_and_score_universe() -> tuple[pd.DataFrame, int]:
    const_path = os.path.join(os.path.dirname(__file__), "constituents.json")
    if not os.path.exists(const_path):
        st.error("❌ `constituents.json` missing.")
        return pd.DataFrame(), 0
    with open(const_path, "r") as f:
        constituents = json.load(f)

    symbols = list(dict.fromkeys([s for syms in constituents.values() for s in syms]))
    tickers = [f"{s}.NS" for s in symbols]
    target_date = get_target_date_key()

    try:
        with st.spinner("🌐 Fetching EOD data for full Nifty 750 universe..."):
            raw = yf.download(tickers, period=HISTORY_PERIOD, group_by="ticker", 
                              threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Yahoo Finance Error: {e}")
        return pd.DataFrame(), 0

    # ✅ FRESHNESS CHECK
    if not _is_yfinance_fresh(raw, target_date):
        return pd.DataFrame(), 0

    # Proceed with scoring
    results = []
    for t in tickers:
        sym = t.replace(".NS", "")
        try:
            sub = raw[t].dropna(how="all") if len(tickers) > 1 else raw.dropna(how="all")
            sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
            res = score_stage2(sub)
            if res:
                res["Symbol"] = sym
                res["Index"] = next((idx for idx, syms in constituents.items() if sym in syms), "Unknown")
                results.append(res)
        except: continue

    df = pd.DataFrame(results)
    return df.sort_values("Score", ascending=False), len(df)

# ──────────────────────────────────────────────
# PURE PYTHON SCORING ENGINE
# ──────────────────────────────────────────────
def _rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def score_stage2(df: pd.DataFrame) -> dict | None:
    if len(df) < 250: return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma50 = c.rolling(50).mean()
    ma150 = c.rolling(150).mean()
    ma200 = c.rolling(200).mean()
    avg_vol_10 = v.rolling(10).mean()
    rsi = _rsi_wilder(c)

    c1, h1, l1, v1 = c.iloc[-1], h.iloc[-1], l.iloc[-1], v.iloc[-1]
    m50, m150, m200 = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]
    r = rsi.iloc[-1]
    vr = v1 / avg_vol_10.iloc[-1] if avg_vol_10.iloc[-1] > 0 else 0
    if np.isnan([m50, m150, m200, vr, r]).any(): return None

    score = 0
    if vr >= 2.0: score += 1
    if h1 >= h.rolling(HH_HL_LOOKBACK).max().shift(1).iloc[-1]: score += 1
    if l1 >= l.rolling(HH_HL_LOOKBACK).min().shift(1).iloc[-1]: score += 1
    if c1 > m50 and ma50.iloc[-1] > ma50.iloc[-MA_RISING_LOOKBACK]: score += 1
    if c1 > m200 and ma200.iloc[-1] > ma200.iloc[-MA_RISING_LOOKBACK]: score += 1
    if c1 > m150: score += 1
    if m50 > m150 > m200: score += 1

    if score >= 6: stage = "🟢 Strong Stage 2"
    elif score >= 4: stage = "🟡 Likely Stage 2"
    elif score >= 2: stage = "🟠 Early/Weak Stage 2"
    else: stage = "⚪ Not Stage 2"

    return {
        "Score": score, "Stage": stage,
        "Illiquid": v1 < MIN_VOLUME,
        "Close": round(c1, 2), "Volume": int(v1), "Vol_Ratio": round(vr, 2),
        "RSI": round(r, 1), "MA50": round(m50, 2), "MA150": round(m150, 2), 
        "MA200": round(m200, 2), "MA_Stack": m50 > m150 > m200
    }

# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="Stage 2 Screener | Nifty 750", page_icon="📈", layout="wide")
st.markdown("""
<style>
.sb-head { font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem; }
.hero { text-align: center; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.2rem; }
.sub-hero { text-align: center; color: #64748b; margin-top: -8px; }
.illiq-tag { color: #dc2626; font-size: 0.7em; font-weight: 800; margin-left: 4px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

def main():
    now_ist = datetime.now(IST).strftime("%d %b %Y · %I:%M %p IST")
    st.markdown('<p class="hero">📊 Nifty Total Market Stage 2 Screener</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-hero">EOD Analysis · 7-Point Weinstein Score · {now_ist}</p>', unsafe_allow_html=True)

    # ── 1. SMART CACHE/FETCH LOGIC (Non-blocking) ──
    target_key = get_target_date_key()
    holidays = load_nse_holidays()
    
    if "full_df" not in st.session_state:
        df = load_json_cache(target_key)
        cache_date = target_key

        if df is None:
            # Cache missing -> ALWAYS attempt fetch
            df, _ = fetch_and_score_universe()
            
            if df is not None and not df.empty:
                save_json_cache(df, target_key)
                cache_date = target_key
            else:
                # Yahoo data stale/unavailable -> Fallback to latest cache
                df, cache_date = load_latest_fallback(target_key)
                if df is None or df.empty:
                    st.warning(f"📅 No cached or fresh EOD data available yet. Try again after 7:30 PM IST.")
                    return
                st.info(f"⚠️ EOD data not yet updated for {target_key}. Showing latest available cache from **{cache_date}**.")
                
        st.session_state["full_df"] = df
        st.session_state["cache_date"] = cache_date

    full_df = st.session_state["full_df"]
    
    # Show soft warning if target date is weekend/holiday but cache exists
    if is_weekend_or_holiday(target_key, holidays) and cache_date != target_key:
        st.info(f"📅 NSE is closed on {target_key}. Displaying cached data from **{cache_date}**.")

    # ── 2. CONTROL PANEL (Checkboxes + Filters) ──
    with st.sidebar.form("controls", clear_on_submit=False):
        st.markdown('<p class="sb-head">🔍 Filters</p>', unsafe_allow_html=True)
        rsi_toggle = st.toggle("Filter: RSI between 50–70", value=False)
        show_illiquid = st.toggle("Show Illiquid Stocks (Vol < 1L)", value=False)
        
        st.markdown("---")
        st.markdown('<p class="sb-head">📦 Select Indices</p>', unsafe_allow_html=True)
        
        const_path = os.path.join(os.path.dirname(__file__), "constituents.json")
        idx_options = list(json.load(open(const_path, "r")).keys()) if os.path.exists(const_path) else []
        
        cols = st.columns(2)
        selected_indices = []
        for i, idx in enumerate(idx_options):
            default_checked = idx in ["Nifty 50", "Nifty Next 50"]
            if cols[i % 2].checkbox(idx, value=default_checked):
                selected_indices.append(idx)
        
        run_btn = st.form_submit_button("🚀 Apply Filters & Show", type="primary", use_container_width=True)

    if "run_triggered" not in st.session_state and run_btn:
        st.session_state["run_triggered"] = True

    if not st.session_state.get("run_triggered"):
        st.info("👈 Select indices/filters and click **Apply Filters & Show** to begin.")
        return

    # ── 3. APPLY FILTERS (Instant) ──
    display_df = full_df.copy()
    if selected_indices: display_df = display_df[display_df["Index"].isin(selected_indices)]
    if rsi_toggle: display_df = display_df[(display_df["RSI"] >= 50) & (display_df["RSI"] <= 70)]
    if not show_illiquid: display_df = display_df[~display_df["Illiquid"]]

    if display_df.empty:
        st.warning("No stocks match the selected filters. Adjust criteria or show illiquid stocks.")
        return

    display_df["Symbol"] = display_df.apply(
        lambda r: f"{r['Symbol']} <span class='illiq-tag'>ILLIQ</span>" if r['Illiquid'] else r['Symbol'], axis=1
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache Date", st.session_state["cache_date"])
    c2.metric("Total Universe", len(full_df))
    c3.metric("Matches (Filters)", len(display_df))
    c4.metric("Strong Stage 2", len(display_df[display_df["Score"] >= 6]))

    def color_rows(row):
        bg_map = {
            "🟢 Strong Stage 2": "#ecfdf5",
            "🟡 Likely Stage 2": "#fefce8",
            "🟠 Early/Weak Stage 2": "#fef2f2",
            "⚪ Not Stage 2": "#f9fafb"
        }
        return [f'background-color: {bg_map.get(row["Stage"], "#ffffff")}'] * len(row)

    styled_df = display_df.style.apply(color_rows, axis=1)

    st.dataframe(
        styled_df, use_container_width=True, hide_index=True, column_config={
            "Symbol": st.column_config.TextColumn("Ticker", width="medium"),
            "Index": st.column_config.TextColumn("Source", width="small"),
            "Stage": st.column_config.TextColumn("Classification", width="medium"),
            "Score": st.column_config.NumberColumn("Score", format="%d/7", width="small"),
            "Close": st.column_config.NumberColumn("Close (₹)", format="%.2f", width="small"),
            "Volume": st.column_config.NumberColumn("Volume", format="%,d", width="medium"),
            "Vol_Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f x", width="small"),
            "RSI": st.column_config.NumberColumn("RSI(14)", format="%.1f", width="small"),
            "Illiquid": None, "Color": None, "MA50": None, "MA150": None, "MA200": None, "MA_Stack": None
        }, height=650
    )

    csv = display_df.drop(columns=["Color"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Screener Results", csv, 
        file_name=f"stage2_screener_{datetime.now(IST).strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True
    )

if __name__ == "__main__":
    main()