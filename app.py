#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   Stage 2 Breakout Screener — Nifty Total Market (750)     ║
║   7-Point Weinstein Scoring | Pure Python Core | EOD Only  ║
║   DATA SEPARATED: constituents.json                        ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime
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
MA_RISING_LOOKBACK = 20      # Change here if needed

# ──────────────────────────────────────────────
# DATA LOADER (PHP-Friendly JSON)
# ──────────────────────────────────────────────
@st.cache_data(ttl=86_400, show_spinner=False)
def load_constituents() -> dict:
    """Load index lists from external JSON. Fails gracefully if missing."""
    path = os.path.join(os.path.dirname(__file__), "constituents.json")
    if not os.path.exists(path):
        st.error("❌ `constituents.json` not found. Please place it in the app directory.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

    if score >= 6: stage, color = "🟢 Strong Stage 2", "#10b981"
    elif score >= 4: stage, color = "🟡 Likely Stage 2", "#f59e0b"
    elif score >= 2: stage, color = "🟠 Early/Weak Stage 2", "#ef4444"
    else: stage, color = "⚪ Not Stage 2", "#6b7280"

    return {
        "Score": score, "Stage": stage, "Color": color,
        "Illiquid": v1 < MIN_VOLUME,
        "Close": round(c1, 2), "Vol_Ratio": round(vr, 2), "RSI": round(r, 1),
        "MA50": round(m50, 2), "MA150": round(m150, 2), "MA200": round(m200, 2),
        "MA_Stack": m50 > m150 > m200
    }

@st.cache_data(ttl=86_400, show_spinner="⏳ Fetching EOD data...")
def run_universe_screener(selected_indices: list[str], rsi_filter: bool = False):
    constituents = load_constituents()
    if not selected_indices or not constituents:
        return pd.DataFrame(), 0
    
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
    valid = 0
    for t in tickers:
        sym = t.replace(".NS", "")
        try:
            sub = raw[t].dropna(how="all") if len(tickers) > 1 else raw.dropna(how="all")
            sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
            res = score_stage2(sub)
            if res:
                valid += 1
                res["Symbol"] = sym
                res["Index"] = idx if any(sym in constituents.get(idx, []) for idx in selected_indices) else "Unknown"
                results.append(res)
        except: continue

    df = pd.DataFrame(results)
    if df.empty: return pd.DataFrame(), valid
    if rsi_filter: df = df[(df["RSI"] >= 50) & (df["RSI"] <= 70)]
    return df.sort_values("Score", ascending=False), valid

# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="Stage 2 Screener | Nifty 750", page_icon="📈", layout="wide")
st.markdown("""
<style>
.sb-head { font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem; }
.hero { text-align: center; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.2rem; }
.sub-hero { text-align: center; color: #64748b; margin-top: -8px; }
.tag { padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; color: #fff; display: inline-block; }
</style>
""", unsafe_allow_html=True)

def main():
    now_ist = datetime.now(IST).strftime("%d %b %Y · %I:%M %p IST")
    st.markdown('<p class="hero">📊 Nifty Total Market Stage 2 Screener</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-hero">EOD Analysis · 7-Point Weinstein Score · {now_ist}</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="sb-head">🎛️ Controls</p>', unsafe_allow_html=True)
        constituents = load_constituents()
        indices = st.multiselect(
            "Select Indices", options=list(constituents.keys()),
            default=["Nifty 50", "Nifty Next 50"],
            help="Combine multiple indices for Nifty 750 coverage"
        )
        rsi_toggle = st.toggle("Filter: RSI between 50–70", value=False)
        show_illiquid = st.toggle("Show Illiquid Stocks (Vol < 1L)", value=False)
        
        st.markdown("---")
        st.caption("ℹ️ Runs on EOD data only. No intra-day refresh.")
        st.caption("🔧 Backend testing available via `python backend_test.py`")

    if not indices:
        st.warning("👈 Select at least one index from the sidebar to begin.")
        return

    df, valid_count = run_universe_screener(indices, rsi_filter=rsi_toggle)
    if df.empty:
        st.info("No stocks matched the criteria with current filters.")
        return

    df_display = df[~df["Illiquid"]] if not show_illiquid else df
    if df_display.empty:
        st.info("All matching stocks are marked Illiquid. Toggle 'Show Illiquid Stocks' to view them.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe Size", len(indices))
    c2.metric("Valid Data", valid_count)
    c3.metric("Stage 2 Matches", len(df))
    c4.metric("Displayed (Liquid)", len(df_display))

    def stage_tag(row):
        return f'<span class="tag" style="background:{row["Color"]}">{row["Stage"]}</span>'
    df_display["Stage"] = df_display.apply(stage_tag, axis=1)
    
    st.dataframe(
        df_display[["Symbol", "Index", "Stage", "Score", "Close", "Vol_Ratio", "RSI", "Illiquid"]],
        use_container_width=True, hide_index=True, column_config={
            "Symbol": st.column_config.TextColumn("Ticker"),
            "Index": st.column_config.TextColumn("Source Index"),
            "Stage": st.column_config.TextColumn("Classification", width="medium"),
            "Score": st.column_config.NumberColumn("Score", format="%d/7"),
            "Close": st.column_config.NumberColumn("Close (₹)", format="%.2f"),
            "Vol_Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f x"),
            "RSI": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
            "Illiquid": st.column_config.CheckboxColumn("Illiquid")
        }, height=600
    )

    csv = df_display.drop(columns=["Stage", "Color"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Screener Results", csv, 
        file_name=f"stage2_screener_{datetime.now(IST).strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True
    )

if __name__ == "__main__":
    main()