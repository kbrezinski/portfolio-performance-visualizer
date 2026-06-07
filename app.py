import os
import certifi
# Ensure SSL env vars are set before importing any network libraries
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Visualizer")


# -----------------------------
# Default portfolios
# -----------------------------
DEFAULT_BENCHMARK = {
    "XIU.TO": 35.0,
    "VFV.TO": 65.0,
}

DEFAULT_CUSTOM = {
    "AVDV": 8.0,
    "AVUV": 8.0,
    "VUN.TO": 25.0,
    "XEC.TO": 10.0,
    "XEF.TO": 20.0,
    "XIC.TO": 29.0,
}

# Defaults for 4 custom portfolios
DEFAULT_CUSTOMS = [
    {},
    {},
    {}
]


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

time_periods = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825,
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(time_periods.keys()),
    index=3
)

# Always show 4 custom portfolios in the top row
num_custom_portfolios = 3

start_date = datetime.today() - timedelta(days=time_periods[selected_period])
end_date = datetime.today()


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=21600)
def fetch_prices(symbols, start_date, end_date):
    """
    Fetch adjusted close price data for a list of symbols.
    Implements chunking and retry/backoff to mitigate 429 (rate limit) errors.
    """
    if not symbols:
        return pd.DataFrame()

    # yfinance can struggle with very large ticker lists — fetch in chunks
    batch_size = 50
    all_prices = pd.DataFrame()

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]

        # retry loop for rate limits
        attempts = 0
        max_attempts = 4
        backoff = 5
        while attempts < max_attempts:
            try:
                data = yf.download(
                    tickers=chunk,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                    timeout=20
                )
                # success
                break
            except Exception as e:
                attempts += 1
                err_text = str(e).lower()
                # If we detect a rate-limit style error, wait and retry with exponential backoff
                if "429" in err_text or "too many" in err_text or "rate limit" in err_text:
                    wait = backoff * (2 ** (attempts - 1))
                    st.warning(f"Rate limit encountered when fetching prices. Retrying in {wait}s (attempt {attempts}/{max_attempts})...")
                    time.sleep(wait)
                    continue
                # For other errors, re-raise after brief wait
                if attempts >= max_attempts:
                    raise
                time.sleep(1)
        else:
            # exhausted retries
            st.error("Failed to fetch market data after multiple attempts — try again later.")
            return pd.DataFrame()

        prices_chunk = pd.DataFrame()

        # If only one symbol is downloaded, yfinance returns a simpler DataFrame
        if len(chunk) == 1:
            sym = chunk[0]
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                prices_chunk[sym] = data["Close"]
        else:
            for sym in chunk:
                try:
                    prices_chunk[sym] = data[sym]["Close"]
                except Exception:
                    # ignore missing symbols in this chunk
                    pass

        # merge chunk results
        if all_prices.empty:
            all_prices = prices_chunk
        else:
            all_prices = all_prices.join(prices_chunk, how="outer")

    all_prices = all_prices.dropna(how="all")
    return all_prices


def calculate_portfolio_returns(portfolio, start_date, end_date, prices_override=None):
    """
    Calculate cumulative portfolio return from weighted symbols.
    Portfolio weights are entered as percentages.
    """
    symbols = [symbol for symbol, weight in portfolio.items() if symbol and weight > 0]

    if not symbols:
        return None

    # Use provided prices DataFrame if available (single fetch for all portfolios)
    if prices_override is not None:
        prices = prices_override.copy()
    else:
        prices = fetch_prices(symbols, start_date, end_date)

    if prices.empty:
        return None

    # Keep only symbols that successfully downloaded
    valid_symbols = [symbol for symbol in symbols if symbol in prices.columns]

    if not valid_symbols:
        return None

    weights = pd.Series(
        {symbol: portfolio[symbol] for symbol in valid_symbols},
        dtype=float
    )

    # Normalize weights to 100%
    weights = weights / weights.sum()

    daily_returns = prices[valid_symbols].pct_change().dropna()

    if daily_returns.empty:
        return None

    portfolio_daily_returns = daily_returns.mul(weights, axis=1).sum(axis=1)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    return cumulative_returns


def portfolio_editor(title, default_portfolio, key_prefix, max_rows=6):
    """
    Simple editable portfolio input area.
    Uses plain text inputs for weights (integers) to avoid spinner controls and decimals.
    """
    st.subheader(title)

    portfolio = {}
    total_weight = 0.0

    default_symbols = list(default_portfolio.keys())
    default_weights = list(default_portfolio.values())

    for i in range(max_rows):
        col1, col2 = st.columns([3, 1])

        default_symbol = default_symbols[i] if i < len(default_symbols) else ""
        default_weight = default_weights[i] if i < len(default_weights) else 0.0

        symbol = col1.text_input(
            f"Symbol {i + 1}",
            value=default_symbol,
            key=f"{key_prefix}_symbol_{i}"
        ).strip().upper()

        # Use a plain text input for integer weights (no spinner arrows, no decimals)
        default_weight_int = int(default_weight) if default_weight else 0
        weight_str = col2.text_input(
            f"Weight %",
            value=str(default_weight_int),
            key=f"{key_prefix}_weight_{i}"
        ).strip()

        try:
            weight = int(weight_str) if weight_str != "" else 0
            if weight < 0:
                weight = 0
        except ValueError:
            weight = 0

        if symbol and weight > 0:
            portfolio[symbol] = weight
            total_weight += weight

    st.write(f"**Total Weight: {total_weight:.0f}%**")

    if total_weight == 0:
        st.warning("Portfolio has no weights.")
    elif abs(total_weight - 100.0) > 0.5:
        st.warning("Weights do not add to 100%. They will be normalized automatically.")

    return portfolio


def total_return(cumulative_returns):
    """
    Calculate total return percentage from cumulative return series.
    """
    if cumulative_returns is None or cumulative_returns.empty:
        return None

    return (cumulative_returns.iloc[-1] - 1) * 100


# -----------------------------
# Portfolio input
# -----------------------------
st.header("Portfolio Configuration")

# Layout: benchmark on the left, 3 custom portfolios across the top row
cols = st.columns([1, 1, 1, 1])

with cols[0]:
    benchmark_portfolio = portfolio_editor(
        "Benchmark",
        DEFAULT_BENCHMARK,
        "benchmark",
        max_rows=6
    )

custom_portfolios = {}
for i in range(num_custom_portfolios):
    with cols[i + 1]:
        title = f"Custom {i + 1}"
        default = DEFAULT_CUSTOMS[i] if i < len(DEFAULT_CUSTOMS) else {}
        custom_portfolios[f"custom_{i}"] = portfolio_editor(
            title,
            default,
            f"custom_{i}",
            max_rows=6
        )


# -----------------------------
# Calculate and plot
# -----------------------------
st.header("Performance")

if "benchmark_returns" not in st.session_state:
    st.session_state.benchmark_returns = None

if "custom_returns" not in st.session_state:
    # store as dict keyed by custom portfolio id (custom_0, custom_1, ...)
    st.session_state.custom_returns = {}

run_update = st.button("Update Chart", type="primary")

if run_update:
    with st.spinner("Loading market data..."):
        # Gather union of all symbols across all portfolios to minimize API calls
        all_symbols = set()
        for p in ([benchmark_portfolio] + list(custom_portfolios.values())):
            for s, w in p.items():
                if s and w > 0:
                    all_symbols.add(s)

        prices = None
        if all_symbols:
            prices = fetch_prices(list(all_symbols), start_date, end_date)

        # Calculate benchmark using single prices DataFrame
        st.session_state.benchmark_returns = calculate_portfolio_returns(
            benchmark_portfolio,
            start_date,
            end_date,
            prices_override=prices
        )

        # calculate each custom portfolio reusing the same prices
        for pid, portfolio in custom_portfolios.items():
            st.session_state.custom_returns[pid] = calculate_portfolio_returns(
                portfolio,
                start_date,
                end_date,
                prices_override=prices
            )

benchmark_returns = st.session_state.benchmark_returns
custom_returns = st.session_state.custom_returns

# Ensure `custom_returns` is a dict. Older runs may have stored a single Series/DataFrame
# directly in session_state.custom_returns — normalize that to a dict keyed by 'custom_0'.
if isinstance(custom_returns, (pd.Series, pd.DataFrame)):
    custom_returns = {"custom_0": custom_returns}
    st.session_state.custom_returns = custom_returns

if custom_returns is None:
    custom_returns = {}
    st.session_state.custom_returns = {}

fig = go.Figure()

if benchmark_returns is not None:
    fig.add_trace(
        go.Scatter(
            x=benchmark_returns.index,
            y=benchmark_returns,
            mode="lines",
            name="Benchmark Portfolio"
        )
    )

# Add traces for all custom portfolios present in session_state
for pid, series in custom_returns.items():
    if series is not None:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=pid.replace("_", " ").title()
            )
        )

if (benchmark_returns is None) and (not any(v is not None for v in custom_returns.values())):
    st.warning("No data available. Check your ticker symbols and try again.")
else:
    fig.update_layout(
        title=f"Portfolio Performance Comparison — {selected_period}",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=600,
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show metrics for benchmark + each custom portfolio
    metrics = []
    metrics.append(("Benchmark Total Return", total_return(benchmark_returns)))

    for i in range(num_custom_portfolios):
        pid = f"custom_{i}"
        metrics.append((f"Custom {i + 1} Total Return", total_return(custom_returns.get(pid))))

    # display metrics in a row (will wrap if too many)
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(
            label,
            "No data" if value is None else f"{value:.2f}%"
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Data from Yahoo Finance via yfinance.")