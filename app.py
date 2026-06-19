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
st.markdown(
    """
    <style>
    .main .block-container{max-width:95%; padding-left:1rem; padding-right:1rem; font-size:16px;}
    html, body, [class*="css"] { font-size:16px; }
    h1, h2, h3 { font-size:1.2rem; }
    /* Increase plotly and Streamlit metric fonts for readability */
    .plotly-graph-div { font-size: 14px; }
    [data-testid="metric-container"] { font-size: 14px; }
    /* Make metric values more prominent when possible */
    [data-testid="metric-container"] .stMetricValue, [data-testid="metric-container"] .metric-value { font-size:18px; font-weight:600; }
    /* Larger buttons for visibility */
    .stButton>button { font-size:18px; padding:14px 24px; width:100%; }
    .stButton>button:hover { transform: translateY(-1px); }
    /* Make headers stand out */
    .main h1, .main h2 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
import yfinance as yf
from streamlit_echarts import st_echarts, JsCode

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
# Optional second benchmark (user's VTI-style benchmark)
# SECOND_BENCHMARK = {"VTI": 48.0, "VXUS": 24.0, "BND": 20.0, "VNQ": 8.0}
# Optional third benchmark
THIRD_BENCHMARK = {"VTI": 20.0, "VXUS": 20.0, "BND": 20.0, "VNQ": 20.0, "GSG": 20.0}
# Optional fourth benchmark
FOURTH_BENCHMARK = {"SPY": 25.0, "VB": 25.0, "VXUS": 25.0, "SHY": 25.0}

# Display-only benchmark breakdowns (category labels + concentrations).
# These are used for the pie charts and do NOT replace the ticker->weight
# benchmark constants above which are used for price fetching and returns.
DEFAULT_BENCHMARK_DISPLAY = {
    "US Equities": 25.0,                 # VUN
    "US Small Cap Value": 8.0,           # AVUV
    "International Small Cap Value": 8.0,# AVDV
    "International Developed": 20.0,     # XEF
    "Emerging Markets": 10.0,            # XEC
    "Canada Equities": 29.0,             # XIC
}

THIRD_BENCHMARK_DISPLAY = {
    "US Equities": 20.0,          # VTI
    "International Equities": 20.0,  # VXUS
    "Bonds": 20.0,                # BND
    "REITs": 20.0,                # VNQ
    "Commodities": 20.0,          # GSG
}

FOURTH_BENCHMARK_DISPLAY = {
    "US Large Cap": 25.0,         # SPY
    "US Small Cap": 25.0,         # VB
    "International Equities": 25.0,  # VXUS
    "Short-Term Bonds/Cash": 25.0,   # SHY
}

# Defaults for 3 custom portfolios (show three editors)
DEFAULT_CUSTOMS = [
    {"TSLA": 14.0, "AAPL": 14.0, "MSFT": 14.0, "GOOGL": 14.0, "AMZN": 14.0,
      "META": 14.0, "NVDA": 16.0},
    {},
    {},
    {}
]



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
# Timeframe and fetch settings (pills selector)
horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "YTD": "ytd",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}

# map human labels to days for slicing
_horizon_days = {
    "1 Months": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "20 Years": 365 * 20,
}

# compute dynamic days for YTD (days since start of current year)
try:
    _horizon_days["YTD"] = (datetime.today() - datetime(datetime.today().year, 1, 1)).days
except Exception:
    _horizon_days["YTD"] = 0

# Sidebar modeled after example.py (query-params enabled)
available_symbols = sorted(set(DEFAULT_BENCHMARK.keys()) | 
                           set(THIRD_BENCHMARK.keys()) | 
                           set(FOURTH_BENCHMARK.keys()) | 
                           {k for d in DEFAULT_CUSTOMS for k in d.keys()})
with st.sidebar:
    st.title(":material/filter_alt: Filters")
    reporting = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="1 Year",
        key="reporting_period",
    )

    y_axis_mode = st.radio("Y-axis", ("Growth of $1", "Percent", "Value ($)"))
    initial_investment = st.number_input("Initial investment ($)", min_value=1, value=1000, step=100)
    benchmarks_sel = st.multiselect(
        "Benchmarks",
        options=["Ken's Benchmark", "Mebane Faber Ivy", "Bill Bernstein"],
        default=["Ken's Benchmark", "Mebane Faber Ivy", "Bill Bernstein"],
        key="benchmarks",
        bind="query-params",
    )
    include_benchmark_ken = "Ken's Benchmark" in benchmarks_sel
    include_benchmark_v3 = "Mebane Faber Ivy" in benchmarks_sel
    include_benchmark_v4 = "Bill Bernstein" in benchmarks_sel
    # Use Streamlit default theme for charts (keep UI stable)
    echarts_theme = "streamlit"

    # expose a small help block
    st.markdown("---")
    st.write("Tip: use the filters above; selections are saved in the URL.")

    # (Removed page-wide theme injection to keep Streamlit default styling)

# map the selected reporting period to the horizon variable used elsewhere
horizon = reporting

slice_start_date = datetime.today() - timedelta(days=_horizon_days.get(horizon, 180))
# Always fetch 5 years of weekly data to allow slicing locally
fetch_end_date = datetime.today()
fetch_start_date = fetch_end_date - timedelta(days=365 * 10)
fetch_interval = '1wk'

# Global toggle removed from sidebar. Controls are defined in the custom sidebar below.


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

# Add a main-page button to expand portfolio editors to 8 rows (one-click)
col_expand_left, _ = st.columns([3,1])
if col_expand_left.button("Show all rows for portfolios (expand to 8)", key="expand_portfolios_main"):
    st.session_state["show_all_portfolios"] = True
    # Attempt a safe rerun that works across Streamlit versions.
    try:
        # Preferred method when available
        st.experimental_rerun()
    except Exception:
        # Fallback: changing query params triggers a rerun in many Streamlit versions
        try:
            st.experimental_set_query_params(_show_all="1")
            # stop execution so the rerun can occur cleanly
        except Exception:
            # Last resort: notify the user to refresh the page
            st.warning("Click Again for the Expanded Rows to Appear.")


# Benchmark is fixed (not shown in editor)
benchmark_portfolio = DEFAULT_BENCHMARK


# -----------------------------
# Calculate and plot
# -----------------------------
st.header("Portfolio Performance Comparison")

if "benchmark_returns" not in st.session_state:
    st.session_state.benchmark_returns = None

if "custom_returns" not in st.session_state:
    # store as dict keyed by custom portfolio id (custom_0, custom_1, ...)
    st.session_state.custom_returns = {}

update_left, _ = st.columns([3,1])
run_update = update_left.button("Update Chart", type="primary", key="update_chart_main")

# Run an initial update on first page load so default-enabled benchmarks are plotted
if "initialized" not in st.session_state:
    st.session_state.initialized = False

do_update = run_update or (not st.session_state.initialized)
if not st.session_state.initialized:
    st.session_state.initialized = True

if do_update:
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
        # include optional benchmark
        if include_benchmark_v3:
            for s, w in THIRD_BENCHMARK.items():
                if s and w > 0:
                    all_symbols.add(s)
                            # include optional benchmark
        if include_benchmark_v4:
            for s, w in FOURTH_BENCHMARK.items():
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

        # Optional benchmark (THIRD)
        if include_benchmark_v3:
            st.session_state.benchmark_returns_v3 = calculate_portfolio_returns(
                THIRD_BENCHMARK,
                fetch_start_date,
                fetch_end_date,
                prices_override=prices
            )
        else:
            st.session_state.benchmark_returns_v3 = None

        # Optional benchmark (FOURTH)
        if include_benchmark_v4:
            st.session_state.benchmark_returns_v4 = calculate_portfolio_returns(
                FOURTH_BENCHMARK,
                fetch_start_date,
                fetch_end_date,
                prices_override=prices
            )
        else:
            st.session_state.benchmark_returns_v4 = None


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
# Descriptive labels for the benchmarks
bench_ken_label = "Ken's Benchmark"
bench_v3_label = "Mebane Faber Ivy"
bench_v4_label = "Bill Bernstein"
# Append benchmarks only if enabled
if st.session_state.get("benchmark_returns_ken") is not None:
    plot_items.append((bench_ken_label, st.session_state.get("benchmark_returns_ken"), 'benchmark_ken'))
if st.session_state.get("benchmark_returns_v3") is not None:
    plot_items.append((bench_v3_label, st.session_state.get("benchmark_returns_v3"), 'benchmark_v3'))
if st.session_state.get("benchmark_returns_v4") is not None:
    plot_items.append((bench_v4_label, st.session_state.get("benchmark_returns_v4"), 'benchmark_v4'))

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
            line=dict(color=color, dash='solid', width=4),
            # show y values with 2 decimal places in hover; include date
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        )
    )
    # store raw `series` as well for metric calculations
    plotted.append((name, color, s, total_pct, annualized, series))
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
        #title={"text": "Portfolio Performance Comparison", "font": {"size": 20}},
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        height=600,
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,  # hide built-in legend, we'll render custom below
        font=dict(size=16),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.25)',
            gridwidth=1,
            title={"text": "Date", "font": {"size": 16}},
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.15)',
            title={"text": yaxis_title, "font": {"size": 16}},
            tickfont=dict(size=14),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Render custom HTML legend for plotted traces just under the plot
    html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px">'
    for name, color, *_ in plotted:
        entry = (
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:40px;height:12px;border-top:3px solid {color};margin-right:8px;'></div>"
                f"<div style='font-size:16px;font-weight:600'>{name}</div>"
            f"</div>"
        )
        html += entry
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Add a clean divider and spacing between legend and metric table
    st.markdown("<hr style='margin-top:18px;margin-bottom:8px'>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Prepare metric values into a list so we can render them as a table
    metrics_data = []
    for item in plotted:
        name, color, series_obj, total_pct, annualized, raw_series = item

        # Final value
        final_display = "N/A"
        try:
            if series_obj is not None and not series_obj.empty:
                last = float(series_obj.iloc[-1])
                if y_axis_mode == "Value ($)":
                    final_display = f"${last:,.2f}"
                elif y_axis_mode == "Growth of $1":
                    final_display = f"{last:.2f}x"
                else:
                    final_display = f"{last:.2f}%"
        except Exception:
            final_display = "N/A"

        delta_text = f"{total_pct:.2f}%" if total_pct is not None else None

        # Variance (display with 2 decimal places)
        var_text = "N/A"
        try:
            if raw_series is not None and not raw_series.empty:
                rets = raw_series.pct_change().dropna()
                if not rets.empty:
                    var_percent = (rets * 100).var()
                    var_text = f"{var_percent:.2f}%"
        except Exception:
            var_text = "N/A"

        # Best and worst calendar years
        worst_text = "N/A"
        best_text = "N/A"
        try:
            if raw_series is not None and not raw_series.empty:
                yearly = raw_series.groupby(raw_series.index.year).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
                if not yearly.empty:
                    worst_year = yearly.idxmin()
                    worst_val = yearly.min() * 100.0
                    worst_text = f"{worst_year}: {worst_val:.2f}%"
                    best_year = yearly.idxmax()
                    best_val = yearly.max() * 100.0
                    best_text = f"{best_year}: {best_val:.2f}%"
        except Exception:
            worst_text = "N/A"
            best_text = "N/A"

        metrics_data.append({
            "name": name,
            "color": color,
            "final": final_display,
            "delta": delta_text,
            "variance": var_text,
            "best": best_text,
            "worst": worst_text,
        })

    # Render per-plot metrics using Streamlit `metric` components in columns
    try:
        stats_cols = st.columns(len(plotted))
        for idx, item in enumerate(plotted):
            name, color, series_obj, total_pct, annualized, raw_series = item
            col = stats_cols[idx]
            # Fixed-height header to keep metric columns vertically aligned.
            # Long names are truncated with ellipsis to avoid pushing metrics out of alignment.
            col.markdown(
                f"<div style='font-size:16px;font-weight:700;color:{color};margin-bottom:6px;min-height:48px;max-height:48px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;display:flex;align-items:center'>{name}</div>",
                unsafe_allow_html=True,
            )

            # Final value display depends on y-axis mode
            final_display = "N/A"
            try:
                if series_obj is not None and not series_obj.empty:
                    last = float(series_obj.iloc[-1])
                    if y_axis_mode == "Value ($)":
                        final_display = f"${last:,.2f}"
                    elif y_axis_mode == "Growth of $1":
                        final_display = f"{last:.2f}x"
                    else:
                        final_display = f"{last:.1f}%"
            except Exception:
                final_display = "N/A"

            delta_text = f"{total_pct:.2f}%" if total_pct is not None else None
            col.metric("Final", final_display, delta=delta_text)

            # Variance of period returns (use raw_series pct_change)
            try:
                var_text = "N/A"
                if raw_series is not None and not raw_series.empty:
                    rets = raw_series.pct_change().dropna()
                    if not rets.empty:
                        var_percent = (rets * 100).var()
                        var_text = f"{var_percent:.1f}%"
                col.metric("Variance", var_text)
            except Exception:
                col.metric("Variance", "N/A")

            # Best calendar-year performance (above worst)
            try:
                best_text = "N/A"
                if raw_series is not None and not raw_series.empty:
                    yearly = raw_series.groupby(raw_series.index.year).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
                    if not yearly.empty:
                        best_year = yearly.idxmax()
                        best_val = yearly.max() * 100.0
                        best_text = f"{best_val:.1f}% ({best_year})"
                col.metric("Best Year", best_text)
            except Exception:
                col.metric("Best Year", "N/A")

            # Worst calendar-year performance
            try:
                worst_text = "N/A"
                if raw_series is not None and not raw_series.empty:
                    yearly = raw_series.groupby(raw_series.index.year).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
                    if not yearly.empty:
                        worst_year = yearly.idxmin()
                        worst_val = yearly.min() * 100.0
                        worst_text = f"{worst_val:.1f}% ({worst_year})"
                col.metric("Worst Year", worst_text)
            except Exception:
                col.metric("Worst Year", "N/A")
    except Exception:
        # fail silently if stats rendering errors
        pass

    # Add a divider then three pie charts side-by-side summarizing allocations
    try:
        st.markdown("<hr style='margin-top:18px;margin-bottom:12px'>", unsafe_allow_html=True)
        st.header("Benchmark Portfolios")

        # Use the display-only benchmark dictionaries so pie charts show
        # the category labels and concentrations (these do not affect
        # portfolio return calculations which still use the ticker maps).
        pie1 = DEFAULT_BENCHMARK_DISPLAY.copy()
        pie2 = THIRD_BENCHMARK_DISPLAY.copy()
        pie3 = FOURTH_BENCHMARK_DISPLAY.copy()

        pie1_title = "Ken's Benchmark"
        pie2_title = "Mebane Faber Ivy"
        pie3_title = "Bill Bernstein"

        c1, c2, c3 = st.columns(3)

        pie1_opts = {
            "title": {"text": pie1_title, "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            #"legend": {"bottom": "0"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": True,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": True, "formatter": "{b}: {d}%"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "14", "fontWeight": "bold"},
                        "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"},
                    },
                    "data": [{"name": k, "value": v} for k, v in pie1.items()],
                }
            ],
        }

        pie2_opts = {
            "title": {"text": pie2_title, "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            #"legend": {"bottom": "0"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": True,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": True, "formatter": "{b}: {d}%"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "14", "fontWeight": "bold"},
                        "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"},
                    },
                    "data": [{"name": k, "value": v} for k, v in pie2.items()],
                }
            ],
        }

        pie3_opts = {
            "title": {"text": pie3_title, "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            #"legend": {"bottom": "0"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": True,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": True, "formatter": "{b}: {d}%"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "14", "fontWeight": "bold"},
                        "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"},
                    },
                    "data": [{"name": k, "value": v} for k, v in pie3.items()],
                }
            ],
        }

        try:
            with c1:
                st_echarts(options=pie1_opts, height="450px", key="pie1", theme=echarts_theme)
            with c2:
                st_echarts(options=pie2_opts, height="450px", key="pie2", theme=echarts_theme)
            with c3:
                st_echarts(options=pie3_opts, height="450px", key="pie3", theme=echarts_theme)
        except Exception:
            # fallback to plotly if echarts fails
            fig1 = px.pie(names=list(pie1.keys()), values=list(pie1.values()), title=pie1_title)
            fig1.update_traces(textposition='inside', textinfo='label+percent')
            fig2 = px.pie(names=list(pie2.keys()), values=list(pie2.values()), title=pie2_title)
            fig2.update_traces(textposition='inside', textinfo='label+percent')
            fig3 = px.pie(names=list(pie3.keys()), values=list(pie3.values()), title=pie3_title)
            fig3.update_traces(textposition='inside', textinfo='label+percent')
            c1.plotly_chart(fig1, use_container_width=True)
            c2.plotly_chart(fig2, use_container_width=True)
            c3.plotly_chart(fig3, use_container_width=True)
    except Exception:
        pass

# -----------------------------
# Bottom: Top Countries by Rank scatter chart
# -----------------------------
try:
    st.markdown("---")
    st.header("Top Countries by Rank")

    # Build DataFrame for the chart
    years = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019]

    blocks = [
        # 2026
        [("South Korea",111.72),("Taiwan",61.78),("Norway",26.7),("Israel",25.14),
         ("Thailand",23.11),("Peru",18.55),("Netherlands",17.55),("Austria",16.13),
         ("Japan",15.14),("Poland",15.12)],

        # 2025
        [("South Korea",95.33),("Peru",86.88),("Spain",78.03),("Poland",77.34),
         ("Greece",76.11),("South Africa",75.2),("Austria",74.2),("Colombia",68.88),
         ("Vietnam",66.55),("Chile",65.41)],

        # 2024
        [("Argentina",63.46),("Israel",34.5),("China",28.98),("United States",23.81),
         ("Singapore",22.1),("Peru",21.72),("Malaysia",19.46),("United Arab Emirates",15.26),
         ("Turkey",12.91),("Taiwan",12.45)],

        # 2023
        [("Argentina",53.65),("Poland",50.7),("Greece",42.69),("Mexico",40.32),
         ("Ireland",35.12),("Brazil",32.6),("Italy",30.64),("Spain",30.26),
         ("Taiwan",29),("United States",26.05)],

        # 2022
        [("Turkey",105.81),("Chile",25.17),("Brazil",12.35),("Argentina",10.36),
         ("Peru",2.13),("Mexico",1.26),("Thailand",1.22),("Greece",0.98),
         ("Indonesia",-0.15),("United Kingdom",-4.38)],

        # 2021
        [("United Arab Emirates",44.1),("Saudi Arabia",33.56),("Austria",31.54),
         ("Taiwan",28.94),("Canada",27),("United States",25.67),("Sweden",22.86),
         ("Israel",22.84),("Netherlands",22.74),("Vietnam",22.05)],

        # 2020
        [("Denmark",42.55),("South Korea",39.44),("Taiwan",31.5),("Netherlands",23.22),
         ("Sweden",22.26),("United States",21.03),("New Zealand",20.04),
         ("Japan",15.41),("India",14.83),("Argentina",14.57)],

        # 2019
        [("Greece",50.2),("Taiwan",33.33),("Netherlands",32.46),("Switzerland",31.58),
         ("United States",30.67),("Colombia",30.4),("New Zealand",30.1),
         ("Ireland",28.14),("Brazil",27.65),("Canada",27.56)]
    ]

    data = []
    for year, block in zip(years, blocks):
        for rank, (country, value) in enumerate(block, start=1):
            data.append([year, rank, country, value])

    df_countries = pd.DataFrame(data, columns=["Year", "Rank", "Country", "Percent"])

    # create a label that includes country name and percent for in-marker text
    df_countries["Label"] = df_countries.apply(lambda r: f"{r['Country']}\n{r['Percent']:.2f}%", axis=1)

    # create a distinct color for each country to avoid reusing similar colors
    unique_countries = sorted(df_countries["Country"].unique())
    try:
        base = px.colors.qualitative.Plotly
        n = len(unique_countries)
        if n <= len(base):
            palette = base[:n]
        else:
            import colorsys
            palette = []
            for i in range(n):
                h = i / max(1, n)
                r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
                palette.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
        color_map = {c: palette[i] for i, c in enumerate(unique_countries)}
    except Exception:
        color_map = None

    fig_countries = px.scatter(
        df_countries,
        x="Year",
        y="Rank",
        color="Country",
        color_discrete_map=color_map,
        hover_data=["Percent"],
        text="Label",
    )

    # enlarge markers, add thin border, and use white text for contrast
    fig_countries.update_traces(
        marker=dict(size=42, symbol="square", line=dict(width=1, color='rgba(0,0,0,0.2)'), opacity=0.95),
        textposition="middle center",
        textfont=dict(color="white", size=11, family="Arial"),
        selector=dict(mode='markers+text')
    )
    # show rank with 1 at top and reverse the year axis so newest -> oldest left-to-right
    fig_countries.update_xaxes(autorange='reversed')
    fig_countries.update_yaxes(autorange="reversed", dtick=1)
    fig_countries.update_layout(
        title="Top Countries by Rank (Color = Country)",
        xaxis_title="Year",
        yaxis_title="Rank (1 = highest)",
        template="plotly_white",
        height=600,
    )

    st.plotly_chart(fig_countries, use_container_width=True)
except Exception:
    st.warning("Failed to render Top Countries by Rank chart.")