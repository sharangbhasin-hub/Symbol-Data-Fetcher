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

# Target timezone (GMT+2), using an IANA timezone
GMT_PLUS_2_TZ = "Europe/Athens"  # example GMT+2 zone[web:170]

# --- Alpha Vantage helpers (equity daily/weekly/monthly) ---

def fetch_alpha_equity(symbol: str, timeframe: str, output_size: str = "full") -> pd.DataFrame:
    """
    Fetch equity time series from Alpha Vantage (daily/weekly/monthly) using documented functions.[file:1]
    """
    tf_map = {
        "daily": "TIME_SERIES_DAILY",
        "weekly": "TIME_SERIES_WEEKLY",
        "monthly": "TIME_SERIES_MONTHLY",
    }
    if timeframe not in tf_map:
        raise ValueError("This app supports daily/weekly/monthly for Alpha Vantage.")

    func = tf_map[timeframe]

    params = {
        "function": func,
        "symbol": symbol,
        "apikey": ALPHA_KEY,
    }
    # Only TIME_SERIES_DAILY supports outputsize; weekly/monthly ignore outputsize per docs.[file:1]
    if timeframe == "daily":
        params["outputsize"] = output_size  # 'compact' or 'full'

    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Detect time series key per docs (e.g. "Time Series (Daily)", "Weekly Time Series").[file:1]
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if ts_key is None:
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")

    ts = data[ts_key]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)

    # Column names in equity time series.[file:1]
    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    }
    df = df.rename(columns=rename_map)
    df = df[[c for c in rename_map.values() if c in df.columns]]
    df = df.astype(float)
    df = df.sort_index()
    return df


def fetch_alpha_equity_range(symbol: str, timeframe: str, start, end) -> pd.DataFrame:
    """Fetch Alpha Vantage equity data and filter between start and end dates."""
    df = fetch_alpha_equity(symbol, timeframe, output_size="full")
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


def fetch_oanda_fx_range_chunked(
    instrument: str,
    granularity: str,
    start,
    end,
    price: str = "M",
    max_candles_per_call: int = 5000,
) -> pd.DataFrame:
    """
    Fetch OANDA candles over a long range by splitting into smaller chunks
    so that each API call stays under the max candles limit.[web:62][web:123]
    """
    seconds_per_candle_map = {
        "M1": 60,
        "M2": 120,
        "M4": 240,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H3": 10800,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "H12": 43200,
        "D": 86400,
        "W": 604800,
        "M": 2592000,  # ~30 days
    }

    if granularity not in seconds_per_candle_map:
        # Fall back to single call if unknown granularity
        return fetch_oanda_fx_range(instrument, granularity, start, end, price=price)

    seconds_per_candle = seconds_per_candle_map[granularity]

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    max_seconds_per_chunk = max_candles_per_call * seconds_per_candle

    chunks = []
    current_start = start_dt

    while current_start < end_dt:
        current_end = current_start + pd.to_timedelta(max_seconds_per_chunk, unit="s")
        if current_end > end_dt:
            current_end = end_dt

        df_chunk = fetch_oanda_fx_range(
            instrument,
            granularity,
            current_start,
            current_end,
            price=price,
        )

        if not df_chunk.empty:
            chunks.append(df_chunk)

        # Move forward by one candle to avoid overlap
        current_start = current_end + pd.to_timedelta(seconds_per_candle, unit="s")

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df_all = pd.concat(chunks)
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    df_all = df_all.sort_index()
    return df_all


# --- Timezone helpers ---

def convert_index_to_gmt_plus_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any DatetimeIndex (UTC or naive) to GMT+2 (e.g. Europe/Athens),
    keeping it tz-aware.[web:166][web:175]
    """
    df2 = df.copy()

    if not isinstance(df2.index, pd.DatetimeIndex):
        return df2

    idx = df2.index

    # If naive, assume UTC then convert to GMT+2
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(GMT_PLUS_2_TZ)
    else:
        idx = idx.tz_convert(GMT_PLUS_2_TZ)

    df2.index = idx
    return df2


def make_timezone_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with any timezone-aware datetimes converted to timezone-naive,
    so Excel can handle them.[web:149][web:157]
    """
    df2 = df.copy()

    # Fix index if it's a DatetimeIndex with tz
    if isinstance(df2.index, pd.DatetimeIndex) and df2.index.tz is not None:
        df2.index = df2.index.tz_localize(None)

    # Fix datetime columns with timezone
    for col in df2.columns:
        if pd.api.types.is_datetime64tz_dtype(df2[col]):
            df2[col] = df2[col].dt.tz_localize(None)

    return df2


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
    # Remove timezone information so Excel writer does not fail.[web:149][web:153]
    df_clean = make_timezone_naive(df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_clean.to_excel(writer, sheet_name=sheet_name)
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
    symbol = st.sidebar.text_input("Equity symbol (e.g. IBM, RELIANCE.BSE)", "IBM")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["daily", "weekly", "monthly"],
        help="Uses TIME_SERIES_DAILY / WEEKLY / MONTHLY as per Alpha Vantage docs.[file:1]",
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
            df = fetch_alpha_equity_range(symbol, timeframe, start_date, end_date)
            # Alpha Vantage times are usually in the market's local timezone (e.g. US/Eastern for US stocks).[file:1][web:173]
            # Treat them as naive local times; localize to US/Eastern, then convert to GMT+2.
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern").tz_convert(GMT_PLUS_2_TZ)
            else:
                df = convert_index_to_gmt_plus_2(df)

        else:
            if not OANDA_KEY:
                st.error("OANDA_API_KEY is not set in environment/.env.")
                st.stop()
            df = fetch_oanda_fx_range_chunked(instrument, granularity, start_date, end_date)
            # OANDA returns times in UTC; convert UTC -> GMT+2.[web:62]
            df = convert_index_to_gmt_plus_2(df)

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
            fname = f"alpha_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.xlsx"
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
