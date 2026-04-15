#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   Stage 2 Breakout Screener — Nifty Total Market (750)     ║
║   7-Point Weinstein Scoring | Persistent Daily Cache       ║
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
# HOLIDAY & WEEKEND CHECKER
# ──────────────────────────────────────────────
def load_nse_holidays() -> set:
    """Parses nse_holidays.json (CM segment) and returns set of YYYY-MM-DD."""
    path = os.path.join(os.path.dirname(__file__), "nse_holidays.json")
    if not os.path.exists(path): return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    holidays = set()
    # CM = Capital Market (Equity). Handle trailing spaces in keys.
    for entry in data.get("CM", []):
        date_str = entry.get("tradingDate ", entry.get("tradingDate", "")).strip()
        try:
            dt = datetime.strptime(date_str, "%d-%b-%Y")
            holidays.add(dt.strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return holidays

def is_market_open(date_str: str, holidays: set) -> bool:
    """Checks if a given YYYY-MM-DD is a weekday and not a holiday."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return date_str not in holidays

# ──────────────────────────────────────────────
# PERSISTENT CACHE UTILS
# ──────────────────────────────────────────────
def get_target_date_key() -> str:
    """<7 PM IST → yesterday, >=7 PM IST → today."""
    now = datetime.now(IST)
    if now.hour >= 19:
        return now.strftime("%Y-%m-%d")
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")

def load_cache(key: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path):
            df = pd.read_json(path, orient="records")
            return df if not df.empty else None
    except Exception:
        return None
    return None

def save_cache(df: pd.DataFrame, key: str):
    path = os.path.join(RESULT_CACHE_DIR, f"{key}.json")
    df.to_json(path, orient="records")

def load_latest_fallback(target_key: str) -> tuple[pd.DataFrame | None, str | None]:
    """Finds the most recent cached file <= target_key."""
    try:
        files = [f.replace(".json", "") for f in os.listdir(RESULT_CACHE_DIR) if f.endswith(".json")]
        files.sort(reverse=True)
        for f in files:
            if f <= target_key:
                return load_cache(f), f
    except Exception:
        pass
    return None, None

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

def run_heavy_calc(selected_indices: list[str], rsi_filter: bool) -> tuple[pd.DataFrame, int]:
    """Fetches EOD data, scores all stocks, returns DF + valid count."""
    path = os.path.join(os.path.dirname(__file__), "constituents.json")
    if not os.path.exists(path):
        st.error("❌ `constituents.json` missing.")
        return pd.DataFrame(), 0
    with open(path, "r") as f:
        constituents = json.load(f)

    symbols = []
    for idx in selected_indices:
        symbols.extend(constituents.get(idx, []))
    symbols = list(dict.fromkeys(symbols))
    tickers = [f"{s}.NS" for s in symbols]

    try:
        raw = yf.download(tickers, period=HISTORY_PERIOD, group_by="ticker", 
                          threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Yahoo Finance Error: {e}")
        return pd.DataFrame(), 0

    results = []
    for t in tickers:
        sym = t.replace(".NS", "")
        try:
            sub = raw[t].dropna(how="all") if len(tickers) > 1 else raw.dropna(how="all")
            sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
            res = score_stage2(sub)
            if res:
                res["Symbol"] = sym
                res["Index"] = next((idx for idx in selected_indices if sym in constituents.get(idx, [])), "Unknown")
                results.append(res)
        except: continue

    df = pd.DataFrame(results)
    if df.empty: return pd.DataFrame(), 0
    if rsi_filter: df = df[(df["RSI"] >= 50) & (df["RSI"] <= 70)]
    return df.sort_values("Score", ascending=False), len(df)

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

    # ── Determine Target Date & Market Status ──
    target_key = get_target_date_key()
    holidays = load_nse_holidays()
    market_open = is_market_open(target_key, holidays)

    # ── CONTROL PANEL (Batched) ──
    with st.sidebar.form("screener_controls", clear_on_submit=False):
        st.markdown('<p class="sb-head">🔍 Filters</p>', unsafe_allow_html=True)
        rsi_toggle = st.toggle("Filter: RSI between 50–70", value=False)
        show_illiquid = st.toggle("Show Illiquid Stocks (Vol < 1L)", value=False)
        
        st.markdown("---")
        st.markdown('<p class="sb-head">📦 Select Indices</p>', unsafe_allow_html=True)
        
        path = os.path.join(os.path.dirname(__file__), "constituents.json")
        idx_options = list(json.load(open(path, "r")).keys()) if os.path.exists(path) else []
        selected_indices = st.multiselect("", idx_options, default=["Nifty 50", "Nifty Next 50"])
        
        run_btn = st.form_submit_button("🚀 Run Screener", type="primary", use_container_width=True)

    # Store only on button press
    if run_btn:
        st.session_state["s_indices"] = selected_indices
        st.session_state["s_rsi"] = rsi_toggle
        st.session_state["s_illiquid"] = show_illiquid

    if "s_indices" not in st.session_state:
        st.info("👈 Select indices and click **Run Screener** to begin.")
        return

    indices = st.session_state["s_indices"]
    if not indices:
        st.warning("Please select at least one index.")
        return

    # ── DATA RESOLUTION LOGIC ──
    df = load_cache(target_key)
    display_date = target_key
    valid_count = len(df) if df is not None else 0

    if df is None:
        if not market_open:
            # Holiday/Weekend: load last available cache
            df, display_date = load_latest_fallback(target_key)
            valid_count = len(df) if df is not None else 0
            if df is None or df.empty:
                st.info(f"📅 Market closed on {target_key}. No cached data available.")
                return
            st.warning(f"📅 Market closed on {target_key}. Showing data from **{display_date}**.")
        else:
            # Trading day, no cache → heavy calculation
            with st.spinner(f"⏳ Fetching & scoring EOD data for {target_key}..."):
                df, valid_count = run_heavy_calc(indices, st.session_state["s_rsi"])
                if not df.empty:
                    save_cache(df, target_key)
                else:
                    st.info("No stocks matched the criteria or data fetch failed.")
                    return

    # ── FILTER & DISPLAY ──
    df_display = df[~df["Illiquid"]] if not st.session_state["s_illiquid"] else df.copy()
    if df_display.empty:
        st.info("All matching stocks are marked Illiquid. Toggle 'Show Illiquid Stocks' to view them.")
        return

    # Format Symbol with ILLIQ tag
    df_display["Symbol"] = df_display.apply(
        lambda r: f"{r['Symbol']} <span class='illiq-tag'>ILLIQ</span>" if r['Illiquid'] else r['Symbol'], axis=1
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Indices", len(indices))
    c2.metric("Valid Data", valid_count)
    c3.metric("Stage 2 Matches", len(df))
    c4.metric("Displayed", len(df_display))

    # Row Coloring
    def color_rows(row):
        bg_map = {
            "🟢 Strong Stage 2": "#ecfdf5", "🟡 Likely Stage 2": "#fefce8",
            "🟠 Early/Weak Stage 2": "#fef2f2", "⚪ Not Stage 2": "#f9fafb"
        }
        return [f'background-color: {bg_map.get(row["Stage"], "#ffffff")}'] * len(row)

    styled_df = df_display.style.apply(color_rows, axis=1)

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

    csv = df_display.drop(columns=["Color"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Screener Results", csv, 
        file_name=f"stage2_screener_{datetime.now(IST).strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True
    )

if __name__ == "__main__":
    main()