import os
import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, date
from dotenv import load_dotenv

# Load environment variables from .env (for local dev)
load_dotenv()

# --- API keys and endpoints ---

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
OANDA_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")

# OANDA base URL per environment (v20 REST API)
OANDA_BASE = (
    "https://api-fxpractice.oanda.com/v3"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com/v3"
)


# --- Alpha Vantage helpers (FX daily/weekly/monthly) ---

def fetch_alpha_fx(pair: str, timeframe: str, output_size: str = "full") -> pd.DataFrame:
    """
    Fetch FX time series from Alpha Vantage.
    Limitations (documented):
    - No 'from'/'to' for FX time series; must request compact/full and filter locally.[web:1]
    - Free key is rate-limited, so large/ frequent requests can be throttled.[web:21][web:88]
    """

    tf_map = {
        "daily": "FX_DAILY",
        "weekly": "FX_WEEKLY",
        "monthly": "FX_MONTHLY",
    }
    if timeframe not in tf_map:
        raise ValueError("For safety, this app supports daily/weekly/monthly for Alpha Vantage.")

    from_symbol, to_symbol = pair.split("/")
    func = tf_map[timeframe]

    params = {
        "function": func,
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": ALPHA_KEY,
        "outputsize": output_size,  # 'compact' or 'full'[web:1]
    }

    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Detect time series key per docs (e.g. "Time Series FX (Daily)")
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if ts_key is None:
        # Typical error when hitting limits is returned in JSON under "Note" or "Error Message"[web:1][web:21]
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")

    ts = data[ts_key]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)

    # FX series typically contain open, high, low, close fields as 1–4.[web:1]
    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
    }
    df = df.rename(columns=rename_map)
    df = df.astype(float)
    df = df.sort_index()
    return df


def fetch_alpha_fx_range(pair: str, timeframe: str, start, end) -> pd.DataFrame:
    """Fetch Alpha Vantage FX data and filter between start and end dates."""
    df = fetch_alpha_fx(pair, timeframe, output_size="full")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    return df.loc[mask]


# --- OANDA helpers (candles with from/to) ---

def to_utc_iso(dt_like) -> str:
    """
    Convert a datetime/date or string to OANDA-compatible UTC ISO8601 (YYYY-MM-DDTHH:MM:SSZ).[web:62]
    """
    dt = pd.to_datetime(dt_like)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt.isoformat().replace("+00:00", "Z")


def fetch_oanda_fx_range(
    instrument: str,
    granularity: str,
    start,
    end,
    price: str = "M",
) -> pd.DataFrame:
    """
    Fetch FX candles from OANDA for a custom date range.
    Uses documented from/to parameters; does not send count with both from and to.[web:62]
    """
    url = f"{OANDA_BASE}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {OANDA_KEY}"}
    params = {
        "granularity": granularity,  # e.g. D, H1, M5 etc. per docs.[web:62]
        "price": price,              # M = mid prices.[web:81]
        "from": to_utc_iso(start),
        "to": to_utc_iso(end),
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    candles = data.get("candles", [])

    records = []
    for c in candles:
        # Skip incomplete candles per docs to avoid partial bars.[web:62]
        if not c.get("complete", False):
            continue
        mid = c.get("mid", {})
        records.append(
            {
                "time": pd.to_datetime(c["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
            }
        )

    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(records).set_index("time").sort_index()
    return df


# --- Data quality checks (for both providers) ---

def quality_checks(df: pd.DataFrame) -> dict:
    """
    Basic quality checks to see if the data is usable for testing.
    Does not assume anything beyond OHLC[V] columns.
    """
    checks = {}
    if df.empty:
        checks["empty"] = True
        return checks

    checks["empty"] = False
    checks["n_rows"] = int(len(df))
    checks["start"] = df.index.min().isoformat()
    checks["end"] = df.index.max().isoformat()

    # Missing values
    checks["n_missing"] = int(df.isna().sum().sum())

    # Non-positive prices
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if price_cols:
        checks["non_positive_prices"] = int((df[price_cols] <= 0).sum().sum())
        # Basic consistency: high >= open/close, low <= open/close
        high_less_than_max = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        low_greater_than_min = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        checks["high_less_than_oc_rows"] = int(high_less_than_max)
        checks["low_greater_than_oc_rows"] = int(low_greater_than_min)
    else:
        checks["non_positive_prices"] = None
        checks["high_less_than_oc_rows"] = None
        checks["low_greater_than_oc_rows"] = None

    # Monotonic time index
    checks["time_monotonic_increasing"] = bool(df.index.is_monotonic_increasing)

    # Gaps: use median step to detect unusually large gaps
    if len(df) > 2:
        deltas = df.index.to_series().diff().dropna()
        median_step = deltas.median()
        large_gaps = (deltas > 2 * median_step).sum()
        checks["large_time_gaps"] = int(large_gaps)
    else:
        checks["large_time_gaps"] = None

    return checks


# --- Excel export (single file) ---

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    """
    Write a DataFrame to an in-memory Excel file and return bytes.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


# --- Streamlit UI ---

st.title("FX Data Fetcher & Quality Check (Alpha Vantage + OANDA)")

st.sidebar.header("1. Data Source")
source = st.sidebar.selectbox("Provider", ["Alpha Vantage", "OANDA"])

# Common date inputs (you can fetch one year at a time if desired)
today = date.today()
default_start = date(today.year - 1, today.month, today.day)

st.sidebar.header("2. Date Range")
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", today)

if start_date > end_date:
    st.sidebar.error("Start date must be before or equal to end date.")

# Source-specific settings
if source == "Alpha Vantage":
    st.sidebar.header("3. Alpha Vantage Settings")
    pair = st.sidebar.text_input("FX pair (FROM/TO)", "EUR/USD")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["daily", "weekly", "monthly"],
        help="Intraday extended is not exposed here because it requires month-slice loops.[web:61][web:82]",
    )
else:
    st.sidebar.header("3. OANDA Settings")
    instrument = st.sidebar.text_input("Instrument", "EUR_USD")
    granularity = st.sidebar.selectbox(
        "Granularity",
        ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "M"],
        help="Granularity must be one of the documented candlestick granularities.[web:62]",
    )

st.sidebar.header("4. Action")
fetch_button = st.sidebar.button("Fetch data")


# Main panel
if fetch_button:
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    try:
        if source == "Alpha Vantage":
            if not ALPHA_KEY:
                st.error("ALPHA_VANTAGE_API_KEY is not set in environment/.env.")
                st.stop()
            df = fetch_alpha_fx_range(pair, timeframe, start_date, end_date)
        else:
            if not OANDA_KEY:
                st.error("OANDA_API_KEY is not set in environment/.env.")
                st.stop()
            df = fetch_oanda_fx_range(instrument, granularity, start_date, end_date)

        st.subheader("Data preview")
        if df.empty:
            st.warning("No data returned for this selection. Check symbol, date range, and API limits.")
        else:
            st.dataframe(df.tail(20))

        st.subheader("Data quality checks")
        qc = quality_checks(df)
        st.json(qc)

        st.subheader("Download as Excel (single file)")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if source == "Alpha Vantage":
            fname = f"alpha_{pair.replace('/','_')}_{timeframe}_{start_date}_{end_date}_{timestamp}.xlsx"
        else:
            fname = f"oanda_{instrument}_{granularity}_{start_date}_{end_date}_{timestamp}.xlsx"

        excel_bytes = df_to_excel_bytes(df)
        st.download_button(
            label="Download Excel file",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error while fetching data: {e}")
        st.stop()
