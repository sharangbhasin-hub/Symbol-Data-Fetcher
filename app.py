import os
import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OANDA_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")

OANDA_BASE = "https://api-fxpractice.oanda.com/v3" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com/v3"


# ---------- Data fetchers ----------

def fetch_alpha_fx(pair: str, timeframe: str, output_size: str = "compact") -> pd.DataFrame:
    tf_map = {
        "1min": ("FX_INTRADAY", "1min"),
        "5min": ("FX_INTRADAY", "5min"),
        "15min": ("FX_INTRADAY", "15min"),
        "30min": ("FX_INTRADAY", "30min"),
        "60min": ("FX_INTRADAY", "60min"),
        "daily": ("FX_DAILY", None),
        "weekly": ("FX_WEEKLY", None),
        "monthly": ("FX_MONTHLY", None),
    }
    from_symbol, to_symbol = pair.split("/")
    func, interval = tf_map[timeframe]

    params = {
        "function": func,
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": ALPHA_KEY,
        "outputsize": output_size,
    }
    if interval:
        params["interval"] = interval

    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Detect time series key
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if ts_key is None:
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")

    ts = data[ts_key]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
        }
    )
    df = df.astype(float)
    df = df.sort_index()
    return df


def fetch_oanda_fx(instrument: str, granularity: str, count: int = 500) -> pd.DataFrame:
    url = f"{OANDA_BASE}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {OANDA_KEY}"}
    params = {
        "granularity": granularity,
        "price": "M",      # mid prices
        "count": count,
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    candles = data.get("candles", [])

    records = []
    for c in candles:
        if not c.get("complete", False):
            continue
        rec = {
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": int(c["volume"]),
        }
        records.append(rec)

    df = pd.DataFrame(records).set_index("time").sort_index()
    return df


# ---------- Data quality checks ----------

def quality_checks(df: pd.DataFrame) -> dict:
    checks = {}
    if df.empty:
        checks["empty"] = True
        return checks

    checks["empty"] = False
    checks["n_rows"] = len(df)
    checks["start"] = df.index.min()
    checks["end"] = df.index.max()

    # Missing values
    checks["n_missing"] = int(df.isna().sum().sum())

    # Non-positive prices
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    checks["non_positive_prices"] = int((df[price_cols] <= 0).sum().sum())

    # Monotonic time index
    checks["time_monotonic_increasing"] = bool(df.index.is_monotonic_increasing)

    # Gaps: count where time difference is > 2x median step
    if len(df) > 2:
        deltas = df.index.to_series().diff().dropna()
        median_step = deltas.median()
        gaps = (deltas > 2 * median_step).sum()
        checks["large_time_gaps"] = int(gaps)
    else:
        checks["large_time_gaps"] = None

    return checks


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Streamlit UI ----------

st.title("FX Data Fetcher (Alpha Vantage & OANDA)")

st.sidebar.header("Data Source")
source = st.sidebar.selectbox("Provider", ["Alpha Vantage", "OANDA"])

if source == "Alpha Vantage":
    st.sidebar.subheader("Alpha Vantage settings")
    pair = st.sidebar.text_input("FX pair (FROM/TO)", "EUR/USD")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"],
    )
    output_size = st.sidebar.selectbox("History length", ["compact", "full"])
else:
    st.sidebar.subheader("OANDA settings")
    instrument = st.sidebar.text_input("Instrument", "EUR_USD")  # OANDA format
    granularity = st.sidebar.selectbox(
        "Granularity",
        ["S5", "S10", "S30", "M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "M"],
    )
    count = st.sidebar.slider("Number of candles", min_value=50, max_value=5000, value=500, step=50)

st.sidebar.header("Actions")
fetch_button = st.sidebar.button("Fetch data")

if fetch_button:
    try:
        if source == "Alpha Vantage":
            df = fetch_alpha_fx(pair, timeframe, output_size)
        else:
            df = fetch_oanda_fx(instrument, granularity, count)

        st.subheader("Preview")
        st.dataframe(df.tail(20))

        st.subheader("Data quality checks")
        qc = quality_checks(df)
        st.json(qc)

        # Download as Excel
        st.subheader("Download")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if source == "Alpha Vantage":
            fname = f"alpha_{pair.replace('/','_')}_{timeframe}_{timestamp}.xlsx"
        else:
            fname = f"oanda_{instrument}_{granularity}_{timestamp}.xlsx"

        excel_bytes = df_to_excel_bytes(df)
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
