import os
import certifi
from pathlib import Path

import streamlit as st
# set wide layout, page icon and small CSS to increase usable width
st.set_page_config(
    page_title="Stock peer analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.markdown("<style>.main .block-container{max-width:95%; padding-left:1rem; padding-right:1rem;}</style>", unsafe_allow_html=True)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
import yfinance as yf

# Set SSL env vars so network libraries use certifi bundle
cert_path = certifi.where()
os.environ["SSL_CERT_FILE"] = cert_path
os.environ["REQUESTS_CA_BUNDLE"] = cert_path
os.environ["CURL_CA_BUNDLE"] = cert_path.replace('\\', '/') if isinstance(cert_path, str) else cert_path

@st.cache_data(ttl=3600)
def fetch_prices_direct(symbols, start_date, end_date, interval='1wk'):
    """
    Fetch adjusted close price series for each symbol using yfinance.Tickers.
    Uses a batch attempt first; falls back to per-symbol with retries/backoff on failures
    (handles transient 429 rate limits). Results cached for 1 hour to avoid repeated
    throttling.
    """
    if not symbols:
        return pd.DataFrame()

    tickers_str = " ".join(symbols)
    collected = {}

    # helper to extract 'Close' series from a history DataFrame
    def extract_close_series(hist_df, sym=None):
        if hist_df is None or hist_df.empty:
            return None
        # MultiIndex columns when multiple tickers
        if isinstance(hist_df.columns, pd.MultiIndex):
            if sym is not None:
                # preferred location ('Close', symbol)
                col = ('Close', sym)
                if col in hist_df.columns:
                    return hist_df[col].dropna().rename(sym)
                # fallback: take xs by ticker then 'Close'
                try:
                    return hist_df.xs(sym, axis=1, level=-1)['Close'].dropna().rename(sym)
                except Exception:
                    return None
            else:
                # caller didn't provide symbol mapping - not used
                return None
        else:
            # single ticker DataFrame
            if 'Close' in hist_df.columns:
                name = sym if sym is not None else symbols[0]
                return hist_df['Close'].dropna().rename(name)
            return None

    def with_retries(fn, *args, **kwargs):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                # treat obvious rate-limit messages specially
                if '429' in msg or 'Too Many Requests' in msg:
                    wait = min(60, (2 ** attempt) * 5)
                    st.warning(f"Rate limited by remote service. Retrying in {wait}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait)
                    continue
                # SSL or network issues: short backoff and retry
                st.warning(f"Transient error fetching data: {e}. Retrying (attempt {attempt+1}/{max_attempts})")
                time.sleep(min(30, 2 ** attempt))
                continue
        return None

    # Batch attempt
    def batch_fetch():
        t = yf.Tickers(tickers_str)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        return t.history(start=start_dt, end=end_dt + pd.Timedelta(days=1), interval=interval, auto_adjust=True)

    hist = with_retries(batch_fetch)
    if hist is not None and not getattr(hist, 'empty', False):
        for s in symbols:
            ser = extract_close_series(hist, s)
            if ser is not None and not ser.empty:
                collected[s] = ser

    # Per-symbol fallback with retries for missing symbols
    missing = [s for s in symbols if s not in collected]
    if missing:
        for s in missing:
            def fetch_one():
                t0 = yf.Ticker(s)
                return t0.history(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval=interval, auto_adjust=True)

            h0 = with_retries(fetch_one)
            ser = extract_close_series(h0, s) if h0 is not None else None
            if ser is not None and not ser.empty:
                collected[s] = ser
            else:
                st.warning(f"{s}: no historical price data or fetch failed (skipped)")

    if not collected:
        return pd.DataFrame()

    prices = pd.concat(collected.values(), axis=1)
    # ensure columns are named correctly (use collected keys order)
    prices.columns = list(collected.keys())
    prices = prices.sort_index()

    # slice to requested range and drop rows where all symbols are NaN
    prices = prices[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
    prices = prices.dropna(how='all')
    return prices


# -----------------------------
# Default portfolios
# -----------------------------

DEFAULT_BENCHMARK = {
    "AVDV": 8.0,
    "AVUV": 8.0,
    "VUN.TO": 25.0,
    "XEC.TO": 10.0,
    "XEF.TO": 20.0,
    "XIC.TO": 29.0,
}

# Defaults for 3 custom portfolios (show three editors)
DEFAULT_CUSTOMS = [
    {"TSLA": 14.0, "AAPL": 14.0, "MSFT": 14.0, "GOOGL": 14.0, "AMZN": 14.0,
      "META": 14.0, "NVDA": 16.0},
    {},
    {},
    {}
]

# Optional second benchmark (user's VTI-style benchmark)
SECOND_BENCHMARK = {"VTI": 48.0, "VXUS": 24.0, "BND": 20.0, "VNQ": 8.0}


# Remove force-refresh and debug buttons — keep a single Update Chart button

# -----------------------------
# Helper UI and calculation functions
# -----------------------------

def portfolio_editor(title, default_portfolio, key_prefix, max_rows=8, visible_rows=4):
    """
    Simple editable portfolio input area.
    Uses plain text inputs for weights (integers) to avoid spinner controls and decimals.
    """
    # Render compact header; row count controlled by the global sidebar toggle
    st.subheader(title)
    # use global session state checkbox to decide how many rows to show for all editors
    show_all = st.session_state.get("show_all_portfolios", False)
    rows_to_show = max_rows if show_all else visible_rows

    portfolio = {}
    total_weight = 0.0

    default_symbols = list(default_portfolio.keys())
    default_weights = list(default_portfolio.values())

    # Render visible rows and still read hidden rows from session state so all max_rows count
    for i in range(max_rows):
        default_symbol = default_symbols[i] if i < len(default_symbols) else ""
        default_weight = default_weights[i] if i < len(default_weights) else 0.0

        default_weight_int = int(default_weight) if default_weight else 0

        # If this row should be shown, render inputs (they will populate session_state).
        if i < rows_to_show:
            col1, col2 = st.columns([3, 1])
            symbol = col1.text_input(
                f"Symbol {i + 1}",
                value=st.session_state.get(f"{key_prefix}_symbol_{i}", default_symbol),
                key=f"{key_prefix}_symbol_{i}"
            ).strip().upper()

            weight_str = col2.text_input(
                f"Weight %",
                value=st.session_state.get(f"{key_prefix}_weight_{i}", str(default_weight_int)),
                key=f"{key_prefix}_weight_{i}"
            ).strip()
        else:
            # Hidden rows: read values that may have been entered previously from session_state
            symbol = st.session_state.get(f"{key_prefix}_symbol_{i}", default_symbol)
            if isinstance(symbol, str):
                symbol = symbol.strip().upper()
            weight_str = str(st.session_state.get(f"{key_prefix}_weight_{i}", str(default_weight_int))).strip()

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


def calculate_portfolio_returns(portfolio, start_date, end_date, prices_override=None):
    """
    Calculate cumulative portfolio return from weighted symbols.
    Portfolio weights are entered as percentages.
    """
    symbols = [symbol for symbol, weight in portfolio.items() if symbol and weight > 0]

    if not symbols:
        return None

    if prices_override is not None:
        prices = prices_override.copy()
    else:
        prices = fetch_prices_direct(symbols, start_date, end_date, interval='1wk')

    if prices.empty:
        return None

    valid_symbols = [s for s in symbols if s in prices.columns]
    if not valid_symbols:
        return None

    weights = pd.Series({s: portfolio[s] for s in valid_symbols}, dtype=float)
    weights = weights / weights.sum()

    returns = prices[valid_symbols].pct_change().dropna()
    if returns.empty:
        return None

    port_daily = returns.mul(weights, axis=1).sum(axis=1)
    cumulative = (1 + port_daily).cumprod()
    return cumulative


# -----------------------------
# Page logic
# -----------------------------
# Always show 3 custom portfolios in the top row
num_custom_portfolios = 4

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

# Timeframe and fetch settings (pills selector)
horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

# map human labels to days for slicing
_horizon_days = {
    "1 Months": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "20 Years": 365 * 20,
}

top_left_cell = st.sidebar.container()
with top_left_cell:
    # Buttons for picking time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="6 Months",
    )

slice_start_date = datetime.today() - timedelta(days=_horizon_days.get(horizon, 180))
# Always fetch 5 years of weekly data to allow slicing locally
fetch_end_date = datetime.today()
fetch_start_date = fetch_end_date - timedelta(days=365 * 5)
fetch_interval = '1wk'

# Y-axis mode and initial investment for value view
y_axis_mode = st.sidebar.radio("Y-axis", ("Growth of $1", "Percent", "Value ($)"))
initial_investment = st.sidebar.number_input("Initial investment ($)", min_value=1, value=1000, step=100)

# Global toggle to expand all portfolio editors (4 rows by default, 8 when checked)
show_all_global = st.sidebar.checkbox("Show all rows for portfolios (expand to 8)", value=False, key="show_all_portfolios")

# Benchmarks: user can opt-in to plot Ken's fixed benchmark and the optional VTI-style benchmark
include_benchmark_ken = st.sidebar.checkbox("Plot Ken's Benchmark", value=False, key="include_benchmark_ken")
include_benchmark_v2 = st.sidebar.checkbox("Plot VTI-style Benchmark", value=False, key="include_benchmark_v2")


# -----------------------------
# Portfolio input
# -----------------------------
st.header("Portfolio Configuration")

# Layout: show only custom portfolios (benchmark is fixed and not editable)
cols = st.columns(num_custom_portfolios)

custom_portfolios = {}
for i in range(num_custom_portfolios):
    with cols[i]:
        title = f"Custom {i + 1}"
        default = DEFAULT_CUSTOMS[i] if i < len(DEFAULT_CUSTOMS) else {}
        # use portfolio_editor defaults (compact view of 4 rows, expand to 8)
        custom_portfolios[f"custom_{i}"] = portfolio_editor(
            title,
            default,
            f"custom_{i}"
        )

# Benchmark is fixed (not shown in editor)
benchmark_portfolio = DEFAULT_BENCHMARK


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
    with st.spinner("Loading market data (direct fetch, 1 month)..."):
        # Collect symbols from custom editors and optionally enabled benchmarks
        all_symbols = set()
        # custom portfolios
        for p in list(custom_portfolios.values()):
            for s, w in p.items():
                if s and w > 0:
                    all_symbols.add(s)
        # include Ken's benchmark only if requested
        if include_benchmark_ken:
            for s, w in DEFAULT_BENCHMARK.items():
                if s and w > 0:
                    all_symbols.add(s)
        # include optional VTI-style benchmark only if requested
        if include_benchmark_v2:
            for s, w in SECOND_BENCHMARK.items():
                if s and w > 0:
                    all_symbols.add(s)

        prices = None
        if all_symbols:
            # Create a deterministic cache key for the requested symbol set and fetch window
            key_symbols = ",".join(sorted(all_symbols))
            fetch_key = f"{key_symbols}|{fetch_start_date.date()}|{fetch_end_date.date()}|{fetch_interval}"
            cached_key = st.session_state.get("last_prices_key")
            cached_prices = st.session_state.get("cached_prices")

            if cached_key == fetch_key and cached_prices is not None:
                # Reuse previously fetched prices without calling the fetcher again
                prices = cached_prices
            else:
                prices = fetch_prices_direct(sorted(all_symbols), fetch_start_date, fetch_end_date, interval=fetch_interval)
                # store in session_state to avoid re-fetching on repeated Update Chart presses
                st.session_state["last_prices_key"] = fetch_key
                st.session_state["cached_prices"] = prices

        # Calculate requested benchmark(s) and custom portfolios using the direct prices
        # Ken's benchmark (optional)
        if include_benchmark_ken:
            st.session_state.benchmark_returns_ken = calculate_portfolio_returns(
                DEFAULT_BENCHMARK,
                fetch_start_date,
                fetch_end_date,
                prices_override=prices
            )
        else:
            st.session_state.benchmark_returns_ken = None

        # Optional VTI-style benchmark
        if include_benchmark_v2:
            st.session_state.benchmark_returns_v2 = calculate_portfolio_returns(
                SECOND_BENCHMARK,
                fetch_start_date,
                fetch_end_date,
                prices_override=prices
            )
        else:
            st.session_state.benchmark_returns_v2 = None

        for pid, portfolio in custom_portfolios.items():
            st.session_state.custom_returns[pid] = calculate_portfolio_returns(
                portfolio,
                fetch_start_date,
                fetch_end_date,
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

# Helper to slice series to the selected view range and convert per y-axis mode
def prepare_series(series):
    if series is None or series.empty:
        return None
    s = series.copy()
    s = s[s.index >= slice_start_date]
    if s.empty:
        return None

    # normalize to the slice start so the view reflects changes from the displayed start
    start_val = float(s.iloc[0])
    if start_val == 0:
        return None

    if y_axis_mode == "Growth of $1":
        return s / start_val
    if y_axis_mode == "Percent":
        return (s / start_val - 1.0) * 100.0
    # Value ($)
    return (s / start_val) * float(initial_investment)

fig = go.Figure()

# build list of (label, series, id) for plotting so we can assign colors consistently
plot_items = []
# Append benchmarks only if enabled
if st.session_state.get("benchmark_returns_ken") is not None:
    plot_items.append(("Ken's Benchmark", st.session_state.get("benchmark_returns_ken"), 'benchmark_ken'))
if st.session_state.get("benchmark_returns_v2") is not None:
    plot_items.append(("Balanced Benchmark", st.session_state.get("benchmark_returns_v2"), 'benchmark_v2'))

for pid, series in custom_returns.items():
    p = custom_portfolios.get(pid)
    tickers = ", ".join(p.keys()) if p else pid.replace("_", " ").title()
    plot_items.append((tickers, series, pid))

# color sequence
colors = px.colors.qualitative.Plotly

plotted = []  # store (name, color, series, total_pct, annualized) for legend and stats
color_idx = 0
for i, (name, series, pid) in enumerate(plot_items):
    s = prepare_series(series)
    if s is None:
        continue
    color = colors[color_idx % len(colors)]
    # compute total and annualized percent over the displayed slice using the original series
    total_pct = None
    annualized = None
    try:
        if series is not None and not series.empty:
            raw_slice = series[series.index >= slice_start_date]
            if not raw_slice.empty:
                start_val = float(raw_slice.iloc[0])
                end_val = float(raw_slice.iloc[-1])
                total_pct = (end_val / start_val - 1.0) * 100.0
                days = (raw_slice.index[-1] - raw_slice.index[0]).days
                if days > 0:
                    annualized = ((end_val / start_val) ** (365.0 / days) - 1.0) * 100.0
    except Exception:
        total_pct = None
        annualized = None

    # always solid
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s,
            mode="lines",
            name=name,
            line=dict(color=color, dash='solid', width=2),
            # show y values with 2 decimal places in hover; include date
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        )
    )
    plotted.append((name, color, s, total_pct, annualized))
    color_idx += 1

# If nothing plotted, show warning
if not plotted:
    st.warning("No data available. Check your ticker symbols and try again.")
else:
    yaxis_title = ""
    if y_axis_mode == "Growth of $1":
        yaxis_title = "Growth of $1"
    elif y_axis_mode == "Percent":
        yaxis_title = "% Return"
    else:
        yaxis_title = "Value ($)"

    fig.update_layout(
        title=f"Portfolio Performance Comparison",
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        height=600,
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,  # hide built-in legend, we'll render custom below
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.25)',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.15)'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Render a compact statistics row under the plot
    try:
        stats_cols = st.columns(len(plotted))
        for idx, item in enumerate(plotted):
            name, color, series_obj, total_pct, annualized = item
            col = stats_cols[idx]
            col.markdown(f"**{name}**")
            if total_pct is None:
                col.write("N/A")
            else:
                col.write(f"Total: {total_pct:.2f}%")
                if annualized is not None:
                    col.write(f"Annualized: {annualized:.2f}% p.a.")
    except Exception:
        # fail silently if stats rendering errors
        pass

    # Render custom HTML legend for plotted traces
    html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px">'
    for name, color, *_ in plotted:
        entry = (
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:40px;height:12px;border-top:3px solid {color};margin-right:8px;'></div>"
            f"<div style='font-size:14px'>{name}</div>"
            f"</div>"
        )
        html += entry
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Bottom metrics: best / worst over the displayed slice
    # Compute best and worst normalized returns (fractional) and show as metrics
    try:
        fractions = []  # list of (fraction, name)
        for name, color, series_obj, total_pct, annualized in plotted:
            if total_pct is None:
                continue
            frac = total_pct / 100.0
            fractions.append((frac, name))

        if fractions:
            max_norm_value = max(fractions, key=lambda x: x[0])
            min_norm_value = min(fractions, key=lambda x: x[0])

            bottom_left_cell = st.container()
            with bottom_left_cell:
                cols = st.columns(2)
                # Show stock name as the main value and delta as percent change
                cols[0].metric(
                    "Best stock",
                    max_norm_value[1],
                    delta=f"{round(max_norm_value[0] * 100, 2)}%",
                    width="content",
                )
                cols[1].metric(
                    "Worst stock",
                    min_norm_value[1],
                    delta=f"{round(min_norm_value[0] * 100, 2)}%",
                    width="content",
                )
    except Exception:
        pass