import os
import certifi
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Set SSL env vars so network libraries use certifi bundle
cert_path = certifi.where()
os.environ["SSL_CERT_FILE"] = cert_path
os.environ["REQUESTS_CA_BUNDLE"] = cert_path
os.environ["CURL_CA_BUNDLE"] = cert_path.replace('\\', '/') if isinstance(cert_path, str) else cert_path

# Sanity check for cert bundle (show actionable warning if missing)
try:
    if not Path(cert_path).exists():
        st.warning(
            "SSL certificate bundle not found at expected path.\n"
            "Run `python -m pip install --upgrade certifi requests urllib3 pyOpenSSL` and restart Streamlit.\n"
            f"Expected cert path: {cert_path}"
        )
except Exception:
    pass

# Debug: show yfinance ticker.info for AAPL (visible in Streamlit UI)
try:
    with st.expander("yfinance debug: AAPL ticker.info", expanded=True):
        st.write("certifi bundle path:", cert_path)
        st.write("SSL_CERT_FILE env:", os.environ.get("SSL_CERT_FILE"))
        st.write("REQUESTS_CA_BUNDLE env:", os.environ.get("REQUESTS_CA_BUNDLE"))
        st.write("CURL_CA_BUNDLE env:", os.environ.get("CURL_CA_BUNDLE"))
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            try:
                st.json(info)
            except Exception:
                st.write(info)
        except Exception as e:
            import traceback
            st.error("Failed to fetch ticker.info for AAPL")
            st.text(str(e))
            st.text(traceback.format_exc())
except Exception:
    pass


# Alpha Vantage settings
# (removed) - switching to yfinance.Tickers for batch fetch


def fetch_prices_direct(symbols, start_date, end_date, interval='1wk'):
    """
    Fetch adjusted close price series for each symbol using yfinance.Tickers.
    Try batch fetch first; if any symbols are missing or the batch fails (SSL or other
    network errors), fall back to per-symbol fetches and warn about skipped tickers.
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

    # Attempt batch fetch first
    try:
        t = yf.Tickers(tickers_str)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        hist = t.history(start=start_dt, end=end_dt + pd.Timedelta(days=1), interval=interval, auto_adjust=True)

        # parse results for each requested symbol
        if hist is not None and not hist.empty:
            for s in symbols:
                ser = extract_close_series(hist, s)
                if ser is not None and not ser.empty:
                    collected[s] = ser
    except Exception as e:
        st.warning(f"yfinance batch fetch failed: {e}")
        hist = None

    # If batch didn't return all symbols, try per-symbol fetch for missing ones
    missing = [s for s in symbols if s not in collected]
    if missing:
        for s in missing:
            try:
                t0 = yf.Ticker(s)
                h0 = t0.history(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval=interval, auto_adjust=True)
                ser = extract_close_series(h0, s)
                if ser is not None and not ser.empty:
                    collected[s] = ser
                else:
                    st.warning(f"{s}: no historical price data (possibly delisted or unsupported)")
            except Exception as e:
                # Common SSL/curl errors may surface here; report and continue
                st.warning(f"Failed to fetch {s}: {e}")

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
    {"TSLA": 50.0, "AAPL": 50.0},
    {},
    {}
]


# Simplify UI: fixed 1-month view (start from scratch)
# Force view to 1M
slice_start_date = datetime.today() - timedelta(days=30)
fetch_start_date = slice_start_date
fetch_end_date = datetime.today()

# Remove force-refresh and debug buttons — keep a single Update Chart button

# -----------------------------
# Helper UI and calculation functions
# -----------------------------

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
num_custom_portfolios = 3

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

# Y-axis mode and initial investment for value view
y_axis_mode = st.sidebar.radio("Y-axis", ("Growth of $1", "Percent", "Value ($)"))
initial_investment = st.sidebar.number_input("Initial investment ($)", min_value=1, value=1000, step=100)

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
        custom_portfolios[f"custom_{i}"] = portfolio_editor(
            title,
            default,
            f"custom_{i}",
            max_rows=6
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
        # Collect symbols from benchmark + custom editors
        all_symbols = set()
        for p in ([benchmark_portfolio] + list(custom_portfolios.values())):
            for s, w in p.items():
                if s and w > 0:
                    all_symbols.add(s)

        prices = None
        if all_symbols:
            prices = fetch_prices_direct(sorted(all_symbols), fetch_start_date, fetch_end_date, interval='1wk')

        # Calculate benchmark and custom portfolios using the direct prices
        st.session_state.benchmark_returns = calculate_portfolio_returns(
            DEFAULT_BENCHMARK,
            fetch_start_date,
            fetch_end_date,
            prices_override=prices
        )

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
if benchmark_returns is not None:
    plot_items.append(("Ken's Benchmark", benchmark_returns, 'benchmark'))

for pid, series in custom_returns.items():
    p = custom_portfolios.get(pid)
    tickers = ", ".join(p.keys()) if p else pid.replace("_", " ").title()
    plot_items.append((tickers, series, pid))

# color sequence
colors = px.colors.qualitative.Plotly

plotted = []  # store (name, color) for legend
color_idx = 0
for i, (name, series, pid) in enumerate(plot_items):
    s = prepare_series(series)
    if s is None:
        continue
    color = colors[color_idx % len(colors)]
    # always solid
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s,
            mode="lines",
            name=name,
            line=dict(color=color, dash='solid', width=2)
        )
    )
    plotted.append((name, color))
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
        showlegend=False  # hide built-in legend, we'll render custom below
    )

    st.plotly_chart(fig, use_container_width=True)

    # Render custom HTML legend for plotted traces
    html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px">'
    for name, color in plotted:
        # draw a small solid line sample and add spacing before the name
        item = (
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:40px;height:12px;border-top:3px solid {color};margin-right:8px;'></div>"
            f"<div style='font-size:14px'>{name}</div>"
            f"</div>"
        )
        html += item
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)