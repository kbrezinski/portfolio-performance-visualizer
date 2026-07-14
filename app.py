import streamlit as st

from streamlit_gtag import st_gtag

# set wide layout, page icon and small CSS to increase usable width
st.set_page_config(
    page_title="Stock peer analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

GA_MEASUREMENT_ID = "G-CTKS1CMJH3"

# Initialize Google Analytics
st_gtag(
    gtag_id="GA_MEASUREMENT_ID",
    config={
        "send_page_view": True
    }
)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode
import time
import yfinance as yf
from streamlit_echarts import st_echarts, JsCode

# -----------------------------
# App configuration
# -----------------------------
FORCE_DARK_THEME = True


# -----------------------------
# Editable benchmark/portfolio constants
# -----------------------------
DEFAULT_BENCHMARK = {
    "AVDV": 8.0,
    "AVUV": 8.0,
    "VUN.TO": 25.0,
    "XEC.TO": 10.0,
    "XEF.TO": 20.0,
    "XIC.TO": 29.0,
    "ZAG.TO": 0.0,
}

SECOND_BENCHMARK = {
    "AVDV": 4.2,
    "AVUV": 7.0,
    "VUN.TO": 21.0,
    "XEC.TO": 5.6,
    "XEF.TO": 11.2,
    "XIC.TO": 21.0,
    "ZAG.TO": 30.0,
}

THIRD_BENCHMARK = {
    "AVDV": 3.0,
    "AVUV": 5.0,
    "VUN.TO": 15.0,
    "XEC.TO": 4.0,
    "XEF.TO": 8.0,
    "XIC.TO": 15.0,
    "ZAG.TO": 50.0,
}

FOURTH_BENCHMARK = {
    "TSLA": 14.0,
    "AAPL": 14.0,
    "MSFT": 14.0,
    "GOOGL": 14.0,
    "AMZN": 14.0,
    "META": 14.0,
    "NVDA": 16.0,
}

BENCHMARK_CONFIG = [
    ("100/0 Benchmark", "benchmark_returns_v1", DEFAULT_BENCHMARK),
    ("70/30 Benchmark", "benchmark_returns_v2", SECOND_BENCHMARK),
    ("50/50 Benchmark", "benchmark_returns_v3", THIRD_BENCHMARK),
    ("Tech Stocks", "benchmark_returns_v4", FOURTH_BENCHMARK),
]
DEFAULT_BENCHMARK_SELECTION = ["100/0 Benchmark", "70/30 Benchmark", "50/50 Benchmark"]

DEFAULT_BENCHMARK_DISPLAY = {
    "US Equities": 25.0,
    "US Small Cap Value": 8.0,
    "International Small Cap Value": 8.0,
    "International Developed": 20.0,
    "Emerging Markets": 10.0,
    "Canada Equities": 29.0,
    "Canadian Bonds": 0.0,
}

DEFAULT_CUSTOMS = [
    {"SPY": 100.0},
    {},
    {},
    {},
]

PROGRESS_GAUGES = [
    {
        "title": "📚 CFP Hours",
        "current": 54,
        "target": 5250,
        "unit": "hrs",
        "color": "#60A5FA",
        "key": "hours",
    },
    {
        "title": "🧑 Clients",
        "current": 10,
        "target": 20,
        "unit": "clients",
        "color": "#10B981",
        "key": "clients",
    },
    {
        "title": "🤝 Meetings",
        "current": 16,
        "target": 100,
        "unit": "meetings",
        "color": "#F97316",
        "key": "meetings",
    },
    {
        "title": "🎓 Completed Courses",
        "current": 3,
        "target": 9,
        "unit": "courses",
        "color": "#14B8A6",
        "key": "courses",
    },
]

def get_chart_theme_colors():
    """Return deterministic chart colors for the forced-dark app theme."""
    return "#f1f5f9", "rgba(255,255,255,0.18)"


def build_portfolio_visualizer_url(portfolio):
    """Build a Portfolio Visualizer backtest URL from a ticker->weight mapping."""
    if not portfolio:
        return None

    valid_items = [(ticker, weight) for ticker, weight in portfolio.items() if ticker and weight > 0]
    if not valid_items:
        return None

    tickers = [ticker for ticker, _ in valid_items]
    weights = [float(weight) for _, weight in valid_items]
    total_weight = sum(weights)
    if total_weight <= 0:
        return None

    # Normalize so allocations are always valid for external backtest params.
    normalized_weights = [round((weight / total_weight) * 100.0, 2) for weight in weights]

    params = {
        "s": "y",
        "initialAmount": 10000,
        "startYear": 1985,
        "benchmark": -1,
        "showYield": "true",
    }

    for i, (ticker, weight) in enumerate(zip(tickers, normalized_weights), start=1):
        params[f"symbol{i}"] = ticker
        params[f"allocation{i}_1"] = weight

    return "https://www.portfoliovisualizer.com/backtest-portfolio?" + urlencode(params)


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


# Remove force-refresh and debug buttons — keep a single Update Chart button

# -----------------------------
# Helper UI and calculation functions
# -----------------------------

def portfolio_editor(title, default_portfolio, key_prefix, max_rows=8, visible_rows=4):
    """
    Simple editable portfolio input area.
    Uses plain text inputs for weights (integers) to avoid spinner controls and decimals.
    """
    # Render simple custom portfolio headers.
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
                f"%",
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
    else:
        visualizer_url = build_portfolio_visualizer_url(portfolio)
        if visualizer_url:
            st.link_button("🔗 Open in Portfolio Visualizer", visualizer_url, use_container_width=True)

        if abs(total_weight - 100.0) > 0.5:
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


def render_return_vs_net_return_section(initial_investment):
    st.header("💸 Return vs Net Return (After Fees)")
    with st.expander("🧠 What this section is showing", expanded=False):
        st.markdown(
            """
            🎯 Think of this as your **30-year return simulator** with two tracks:

            🚀 **Gross line**: growth before fees and behavioural drag.

            🧾 **Net line**: growth after fees and selected behavioural adjustments.

                🔵 Blue box - Scenario Controls: tune return, fees, and starting investment.

                🟢 Green box - Typical Assumptions: quick ranges you can copy into the controls.

                🟡 Yellow box - Behavioural Adjustments: turn on real-world frictions (they subtract from return).

                🔴 Red box - Behavioural Examples: reference ranges for common behavioural effects.

            💡 Pro tip: try one toggle at a time to see which friction hurts long-term compounding the most.
            """
        )

    top_left, top_right = st.columns(2)

    with top_left:
        st.info("🎛️ Scenario Controls")

        ctrl_a, ctrl_b = st.columns(2)

        with ctrl_a:
            gross_return_pct = st.slider(
                "Annual Return (%)",
                min_value=0.0,
                max_value=15.0,
                value=8.0,
                step=0.1,
                format="%.1f%%",
                key="fee_proj_gross_return",
            )

        with ctrl_b:
            fund_fee_pct = st.slider(
                "Fund Fee or MER (%)",
                min_value=0.0,
                max_value=2.5,
                value=1.0,
                step=0.01,
                format="%.2f%%",
                key="fee_proj_fund_fee",
            )

        ctrl_c, ctrl_d = st.columns(2)
        with ctrl_c:
            mgmt_fee_pct = st.slider(
                "Management Fee (%)",
                min_value=0.0,
                max_value=2.5,
                value=1.0,
                step=0.25,
                format="%.2f%%",
                key="fee_proj_mgmt_fee",
            )

        with ctrl_d:
            fee_projection_initial = st.number_input(
                "Starting investment ($)",
                min_value=1,
                value=int(initial_investment),
                step=100_000,
                key="fee_projection_initial",
            )
            st.caption("Timeframe: 30 years")

        sample_rows = [
            {"": "Annual Returns", "%": "4.0 - 12.0% (**7.12%** is typical"},
            {"": "Fund Fee or MER", "%": "0.1 - 1.0%"},
            {"": "Management Fee", "%": "0.5 - 1.0%"},
        ]
        sample_df = pd.DataFrame(sample_rows)
        st.subheader("📌 Sample Typical Assumptions")
        st.table(sample_df, hide_header=True)

    with top_right:
        st.warning("🧩 Behavioural Adjustments (toggle on to apply)")

        behavior_defaults = {
            "No Rebalancing": 0.25,
            "Behavioural Gap": 1.5,
            "Tax Optimization": 0.40,
            "Asset Location": 0.50,
        }

        b1, b2 = st.columns(2)
        with b1:
            apply_no_rebal = st.checkbox(
                f"No Rebalancing (-{behavior_defaults['No Rebalancing']:.2f}%)",
                value=False,
                key="behavior_no_rebal",
            )
            apply_tax_opt = st.checkbox(
                f"Tax Optimization (-{behavior_defaults['Tax Optimization']:.2f}%)",
                value=False,
                key="behavior_tax_opt",
            )
        with b2:
            apply_behavioral_gap = st.checkbox(
                f"Behavioural Gap (-{behavior_defaults['Behavioural Gap']:.2f}%)",
                value=False,
                key="behavior_gap",
            )
            apply_asset_location = st.checkbox(
                f"Asset Location (-{behavior_defaults['Asset Location']:.2f}%)",
                value=False,
                key="behavior_asset_loc",
            )

        behavior_drag_pct_total = (
            (behavior_defaults["No Rebalancing"] if apply_no_rebal else 0.0)
            + (behavior_defaults["Behavioural Gap"] if apply_behavioral_gap else 0.0)
            + (behavior_defaults["Tax Optimization"] if apply_tax_opt else 0.0)
            + (behavior_defaults["Asset Location"] if apply_asset_location else 0.0)
        )
        st.caption(f"🎯 Total behavioral adjustment applied: -{behavior_drag_pct_total:.2f}%")
        if behavior_drag_pct_total > 0 and mgmt_fee_pct > 0:
            st.warning(
                "Heads up 👋 You currently have behavioural adjustments and a non-zero management fee turned on. "
                "Some of that impact may overlap, so your net result could be a bit conservative."
            )

        sample_rows = [
            {"Typical Case": "No Reblancing", "%": "-0.1 - 0.4%"},
            {"Typical Case": "Behavioural Gap (Picking individual stocks)", "%": "-1.0 - 2.0%"},
            {"Typical Case": "Tax Optimization (Tax Advice)", "%": "-0.3 - 0.6%"},
            {"Typical Case": "Asset Location (Efficent tax accounts)", "%": "-0.3 - 0.75%"},
        ]
        sample_df = pd.DataFrame(sample_rows)
        st.subheader("🧪 Sample Behavioural Assumptions")
        sample_df = sample_df.style.map(lambda x: "color: #ff4d4d;", subset=["%"])
        st.table(sample_df, hide_header=True)

    st.divider()

    projection_years = 30
    timeline_years = list(range(0, projection_years + 1))

    gross_rate = gross_return_pct / 100.0
    behavior_drag_rate = behavior_drag_pct_total / 100.0
    effective_gross_rate = max(gross_rate - behavior_drag_rate, -0.99)
    fund_fee_rate = fund_fee_pct / 100.0
    mgmt_fee_rate = mgmt_fee_pct / 100.0

    fee_only_rate = ((1 + gross_rate) * (1 - fund_fee_rate) * (1 - mgmt_fee_rate) - 1)
    net_rate = ((1 + effective_gross_rate) * (1 - fund_fee_rate) * (1 - mgmt_fee_rate) - 1)

    baseline_curve_value = [float(fee_projection_initial) * ((1 + gross_rate) ** yr) for yr in timeline_years]
    fee_only_curve_value = [float(fee_projection_initial) * ((1 + fee_only_rate) ** yr) for yr in timeline_years]
    net_curve_value = [float(fee_projection_initial) * ((1 + net_rate) ** yr) for yr in timeline_years]

    fee_fig = go.Figure()
    fee_fig.add_trace(
        go.Scatter(
            x=timeline_years,
            y=baseline_curve_value,
            mode="lines",
            name="Baseline gross (no fees, no behavioural drag)",
            line=dict(width=4),
            hovertemplate="Year %{x}: $%{y:,.2f}<extra></extra>",
        )
    )
    fee_fig.add_trace(
        go.Scatter(
            x=timeline_years,
            y=fee_only_curve_value,
            mode="lines",
            name="After fees only",
            line=dict(width=4, dash="dot"),
            hovertemplate="Year %{x}: $%{y:,.2f}<extra></extra>",
        )
    )
    fee_fig.add_trace(
        go.Scatter(
            x=timeline_years,
            y=net_curve_value,
            mode="lines",
            name="After fees + behavioural drag (all-in)",
            line=dict(width=4, dash="dash"),
            hovertemplate="Year %{x}: $%{y:,.2f}<extra></extra>",
        )
    )

    fee_fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Portfolio value ($)",
        height=520,
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=16),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.25)',
            gridwidth=1,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.15)',
            tickfont=dict(size=14),
            tickformat=",.0f",
            tickprefix="$",
        ),
    )

    gross_end_value = float(fee_projection_initial) * ((1 + gross_rate) ** projection_years)
    fee_only_end_value = float(fee_projection_initial) * ((1 + fee_only_rate) ** projection_years)
    net_end_value = float(fee_projection_initial) * ((1 + net_rate) ** projection_years)

    fee_cost_value = gross_end_value - fee_only_end_value
    behavioral_opp_cost_value = fee_only_end_value - net_end_value

    gross_app_pct = (gross_end_value / float(fee_projection_initial) - 1.0) * 100.0
    net_app_pct = (net_end_value / float(fee_projection_initial) - 1.0) * 100.0
    fee_cost_pct = (fee_only_end_value / gross_end_value - 1.0) * 100.0 if gross_end_value != 0 else 0.0
    behavioral_opp_cost_pct = (net_end_value / fee_only_end_value - 1.0) * 100.0 if fee_only_end_value != 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Ending value (gross)", f"${gross_end_value:,.2f}", delta=f"{gross_app_pct:.2f}%")
    c2.metric("Ending value (net)", f"${net_end_value:,.2f}", delta=f"{net_app_pct:.2f}%")
    c3.metric(
        "Lost to fund + management fees",
        f"${fee_cost_value:,.2f}",
        delta=f"{fee_cost_pct:.2f}% vs no-fee baseline"
    )
    c4.metric(
        "Additional loss from behavioural drag",
        f"${behavioral_opp_cost_value:,.2f}",
        delta=f"{behavioral_opp_cost_pct:.2f}% vs fees-only"
    )

    st.plotly_chart(fee_fig, use_container_width=True)


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
with st.sidebar:
    try:
        from streamlit_extras.avatar import avatar
    except ImportError:  # pragma: no cover - optional dependency fallback
        avatar = None

    try:
        from streamlit_extras.buy_me_a_coffee import button
    except ImportError:  # pragma: no cover - optional dependency fallback
        button = None

    avatar_image = "avatar.jpeg"
    if avatar is not None:
        avatar(avatar_image, height=100, border=True,
               label="Kenneth Brezinski",
               caption="Financial Advisor Student",)
    else:
        st.image(avatar_image, width=60)

    if button is not None:
        button(username="fake-username", floating=False, width=221)
    else:
        st.caption("Buy Me a Coffee button unavailable")

    st.title("🔎 Filters")
    reporting = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="1 Year",
        key="reporting_period",
    )

    y_axis_mode = st.radio("Y-axis", ("Growth of $1", "Percent", "Value ($)"))
    initial_investment = st.number_input("Initial investment ($)", min_value=1, value=100_000, step=10_000)
    benchmarks_sel = st.multiselect(
        "Benchmarks",
        options=[label for label, _, _ in BENCHMARK_CONFIG],
        default=DEFAULT_BENCHMARK_SELECTION,
        key="benchmarks",
        bind="query-params",
    )
    selected_benchmarks = set(benchmarks_sel)
    echarts_theme = "dark"

    # expose a small help block
    st.markdown("---")
    st.write("💡 Tip: use the filters above; selections are saved in the URL.")

    # ---------- Reusable Gauge ----------
    def render_progress_gauge(title, current, target, unit, color, key):

        pct = (current / target) * 100

        # Use deterministic colors to avoid browser/theme mismatch.
        text_color, _ = get_chart_theme_colors()

        option = {
            "title": {
                "text": title,
                "left": "center",
                "top": "2%",
                "textStyle": {
                    "color": text_color,
                    "fontSize": 16,
                    "fontWeight": "bold",
                },
            },

            "series": [
                {
                    "type": "gauge",

                    "center": ["50%", "60%"],

                    "min": 0,
                    "max": 100,

                    "startAngle": 90,
                    "endAngle": -270,

                    "pointer": {"show": False},
                    "progress": {"show": True, "roundCap": True, "width": 12, "itemStyle": {"color": color}},
                    "axisLine": {"lineStyle": {"width": 12, "color": [[1, "#E5E7EB"]]}},
                    "axisTick": {"show": False},
                    "splitLine": {"show": False},
                    "axisLabel": {"show": False},
                    "title": {"show": False},
                    "detail": {
                        "valueAnimation": True,
                        "formatter": "{value}%",
                        "offsetCenter": [0, "0%"],
                        "fontSize": 30,
                        "fontWeight": "bold",
                        "color": text_color,
                    },

                    "data": [{"value": round(pct, 1)}],
                }
            ],

            "graphic": [
                {
                    "type": "text",
                    "left": "center",
                    "top": "70%",
                    "style": {
                        "text": f"{current:,} / {target:,} {unit}",
                        "fill": color,
                        "fontSize": 12,
                        "fontWeight": 500,
                    },
                }
            ],
        }

        st_echarts(
            options=option,
            height="190px",
            key=f"gauge_{key}",
        )


    # ---------- Progress Section ----------
    with st.expander("📈 Professional Progress towards CFP Designation", expanded=False):
        st.caption("As of July 12th, 2026")

        for gauge in PROGRESS_GAUGES:
            render_progress_gauge(
                title=gauge["title"],
                current=gauge["current"],
                target=gauge["target"],
                unit=gauge["unit"],
                color=gauge["color"],
                key=gauge["key"],
            )

        st.markdown("""
            **Courses Completed (3/9):**  
            ✅ Financial Management  
            ✅Tax Planning  
            ✅Investment Planning  
            """)

    slice_start_date = datetime.today() - timedelta(days=_horizon_days.get(reporting, 180))
    # Always fetch 5 years of weekly data to allow slicing locally
    fetch_end_date = datetime.today()
    fetch_start_date = fetch_end_date - timedelta(days=365 * 10)
    fetch_interval = '1wk'

    # Global toggle removed from sidebar. Controls are defined in the custom sidebar below.

main_tab, returns_tab, references_tab = st.tabs(
    ["📋 Portfolio Configuration", "💸 Return vs Net Return", "🔗 References & Calculators"]
)

with returns_tab:
    render_return_vs_net_return_section(initial_investment)

with references_tab:
    st.info("Links marked with 🐐 are Ken's personal favorites 💖")
    st.header("🔗 External References")
    st.markdown(
        """
        - [🐐 Portfolio Visualizer](https://www.portfoliovisualizer.com/backtest-portfolio?s=y&allocation4_1=30&initialAmount=10000&showYield=true&allocation1_1=40&allocation3_1=10&allocation2_1=20&symbol4=VBMFX&startYear=1985&symbol1=VTSMX&symbol2=VGTSX&symbol3=VGSIX&benchmark=-1&benchmarkSymbol=VFINX)
        - [🐐 ETF Selector (ETFDb)](https://etfdb.com/etfs/)
        """
    )

    st.header("🧮 Useful Tools")
    st.markdown(
        """
        - [Rent Vs Buy (PWL Capital)](https://research-tools.pwlcapital.com/research/rent-vs-buy)
        - [🐐 Portfolio Rebalancer (Passiv)](https://app.passiv.com/login?next=%2Freporting)
        """
    )

# -----------------------------
# Portfolio input
# -----------------------------
with main_tab:
    st.header("📋 Portfolio Configuration")

    cols = st.columns([1.2, 1.2, 1.2, 1.2, 3.2])

    custom_portfolios = {}

    for i in range(num_custom_portfolios):
        with cols[i]:
            custom_titles = [
                "🟢 Portfolio 1",
                "🔵 Portfolio 2",
                "🟠 Portfolio 3",
                "🔴 Portfolio 4",
            ]
            title = custom_titles[i] if i < len(custom_titles) else f"Custom Portfolio {i + 1}"
            default = DEFAULT_CUSTOMS[i] if i < len(DEFAULT_CUSTOMS) else {}

            custom_portfolios[f"custom_{i}"] = portfolio_editor(
                title,
                default,
                f"custom_{i}"
            )

    with cols[4]:
        st.markdown("## 🎉 Build Your Portfolio Like a Pro")
        st.success("✨ Quick playbook: add symbols, set weights, and launch your visualizer link.")
        st.markdown(
            """
            ### 🧭 How to use this page

            💼 **Step 1:** Add ticker symbols and % weights in each custom portfolio.

            🇨🇦 **Step 2:** If a symbol fails, try adding `.TO` (example: `VUN.TO`).

            🧩 **Step 3:** Need more holdings? Click **Show all rows for portfolios** to expand from 4 to 8 rows.

            🔗 **Step 4:** Once a portfolio has weights, click **Open in Portfolio Visualizer** to test that mix on the external site.

            🚀 **Step 5:** Click **Update Portfolio Charts** to refresh all charts and metrics.
            """
        )
        st.warning("🎯 Friendly reminder: weights are auto-normalized, so they do not have to total exactly 100%.")

    if not st.session_state.get("show_all_portfolios", False):
        button_left, button_right = st.columns([4.8, 3.2])
        with button_left:
            if st.button("🧩 Show all rows for portfolios (expand to 8)", key="expand_portfolios_main", use_container_width=True, type="secondary"):
                st.session_state["show_all_portfolios"] = True
                st.rerun()


with main_tab:
    # -----------------------------
    # Calculate and plot
    # -----------------------------
    st.markdown("---")
    st.header("📈 Portfolio Performance Comparison")

    perf_left, perf_right = st.columns([2, 1])

    with perf_right:
        st.markdown("## 🧠 Chart Game Plan")
        st.success("📊 Pick your view, compare like a detective, then update to lock in changes.")
        st.markdown(
            """
            🗓️ **Time horizon:** change it in the sidebar to zoom in or out.

            🎛️ **Y-axis mode:** switch between **Growth**, **Percent**, and **Value ($)**.

            ⚖️ **Benchmarks:** select one or more to compare against your custom mixes.

            🖱️ **Hover mode:** move your cursor over the chart to compare all lines on the same date.

            🚀 **After edits:** click **Update Portfolio Charts** to refresh the chart and metrics.
            """
        )
        st.warning("✨ Pro tip: try a short horizon first, then expand to 5Y or 10Y for bigger trend context.")

    if "benchmark_returns" not in st.session_state:
        st.session_state.benchmark_returns = None

    if "custom_returns" not in st.session_state:
        # store as dict keyed by custom portfolio id (custom_0, custom_1, ...)
        st.session_state.custom_returns = {}

    with perf_left:
        st.info("📌 After changing tickers, weights, benchmarks, or chart settings, click **Update Portfolio Charts** below.")
        run_update = st.button(
            "🚀 Update Portfolio Charts ⭐",
            type="primary",
            use_container_width=True,
            key="update_chart_main",
        )

    # Run an initial update on first page load so default-enabled benchmarks are plotted
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    do_update = run_update or (not st.session_state.initialized)
    if not st.session_state.initialized:
        st.session_state.initialized = True

    if do_update:
        with st.spinner("Loading market data (direct fetch, 1 month)..."):
            all_symbols = set(s for p in custom_portfolios.values() for s, w in p.items() if s and w > 0)
            for label, _, portfolio in BENCHMARK_CONFIG:
                if label in selected_benchmarks:
                    all_symbols.update(s for s, w in portfolio.items() if s and w > 0)

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

            for label, state_key, portfolio in BENCHMARK_CONFIG:
                st.session_state[state_key] = (
                    calculate_portfolio_returns(portfolio, fetch_start_date, fetch_end_date, prices_override=prices)
                    if label in selected_benchmarks else None
                )

            for pid, portfolio in custom_portfolios.items():
                st.session_state.custom_returns[pid] = calculate_portfolio_returns(
                    portfolio,
                    fetch_start_date,
                    fetch_end_date,
                    prices_override=prices
                )

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

    plot_items = []
    for label, state_key, _ in BENCHMARK_CONFIG:
        series = st.session_state.get(state_key)
        if series is not None:
            plot_items.append((label, series, state_key))

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
        with perf_left:
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

        with perf_left:
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
                col.metric("Final", final_display, delta=delta_text, border=True)

                # Variance: average of year-over-year variance with 10-year trend
                try:
                    var_text = "N/A"
                    var_delta_text = None
                    variance_spark_10 = None
                    if raw_series is not None and not raw_series.empty:
                        rets = raw_series.pct_change().dropna()
                        if not rets.empty:
                            # Year-over-year variance based on the full fetched return history.
                            yearly_variance = (rets * 100).groupby(rets.index.year).var().dropna()
                            if not yearly_variance.empty:
                                avg_variance = yearly_variance.mean()
                                var_text = f"{avg_variance:.1f}%"
                                var_delta_text = f"+{avg_variance:.1f}%"
                                variance_spark_10 = [round(v, 1) for v in yearly_variance.tail(10).tolist()]

                    col.metric(
                        "Avg. Variance",
                        var_text,
                        delta=var_delta_text,
                        border=True,
                        chart_data=variance_spark_10,
                        chart_type="area",
                    )
                except Exception:
                    col.metric("Avg. Variance", "N/A", border=True)

                # Best calendar-year performance (above worst)
                try:
                    best_year_value = "N/A"
                    best_delta_text = None
                    yearly_spark_10 = None
                    if raw_series is not None and not raw_series.empty:
                        yearly = raw_series.groupby(raw_series.index.year).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
                        if not yearly.empty:
                            best_year = yearly.idxmax()
                            best_val = yearly.max() * 100.0
                            best_year_value = str(int(best_year))
                            best_delta_text = f"{best_val:+.1f}%"

                            # Show the full fetched history trend as 10 yearly values.
                            yearly_spark_10 = [round(v, 1) for v in (yearly * 100.0).tail(10).tolist()]

                    col.metric(
                        "Best Year",
                        best_year_value,
                        delta=best_delta_text,
                        border=True,
                        chart_data=yearly_spark_10,
                        chart_type="bar",
                    )
                except Exception:
                    col.metric("Best Year", "N/A", border=True)

                # Worst drawdown year (yearly returns capped at 0)
                try:
                    worst_drawdown_year = "N/A"
                    worst_drawdown_delta = None
                    drawdown_spark_10 = None
                    if raw_series is not None and not raw_series.empty:
                        yearly = raw_series.groupby(raw_series.index.year).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1.0)
                        if not yearly.empty:
                            # Positive years are 0 drawdown; negatives remain negative.
                            yearly_drawdown = yearly.apply(lambda r: min(r, 0.0))

                            worst_year = yearly_drawdown.idxmin()
                            worst_val = yearly_drawdown.min() * 100.0
                            worst_drawdown_year = str(int(worst_year))

                            # Keep a negative sign (including -0.0%) to preserve red delta styling intent.
                            worst_drawdown_delta = f"-{abs(worst_val):.1f}%"
                            drawdown_spark_10 = [round(v, 1) for v in (yearly_drawdown * 100.0).tail(10).tolist()]

                    col.metric(
                        "Worst Drawdown Year",
                        worst_drawdown_year,
                        delta=worst_drawdown_delta,
                        border=True,
                        chart_data=drawdown_spark_10,
                        chart_type="area",
                    )
                except Exception:
                    col.metric("Worst Drawdown Year", "N/A", border=True)
        except Exception:
            # fail silently if stats rendering errors
            pass

        # Add a divider then pie chart for Ken's Benchmark
        # (THIRD_BENCHMARK_DISPLAY and FOURTH_BENCHMARK_DISPLAY are available as placeholders for future asset mix)
        try:
            st.markdown("<hr style='margin-top:18px;margin-bottom:12px'>", unsafe_allow_html=True)
            st.header("🏦 Ken's Benchmark Portfolio")

            # Use the display-only benchmark dictionaries so pie charts show
            # the category labels and concentrations (these do not affect
            # portfolio return calculations which still use the ticker maps).
            pie1 = DEFAULT_BENCHMARK_DISPLAY.copy()
            pie1_title = "Asset Allocation"
            label_color, _ = get_chart_theme_colors()

            pie1_opts = {
                "backgroundColor": "transparent",
                "title": {
                    "text": "Asset Allocation",
                    "left": "center",
                    "textStyle": {"color": label_color},
                },
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                #"legend": {"bottom": "0"},
                "series": [
                    {
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": True,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "rgba(15,23,42,0.7)",
                            "borderWidth": 2,
                        },
                        "label": {
                            "show": True,
                            "formatter": "{b}: {d}%",
                            "color": label_color,
                        },
                        "emphasis": {
                            "label": {"show": True, "fontSize": "14", "fontWeight": "bold"},
                            "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"},
                        },
                        "data": [{"name": k, "value": v} for k, v in pie1.items()],
                    }
                ],
            }

            try:
                row1_left, row1_right = st.columns(2)
                with row1_left:
                    st_echarts(options=pie1_opts, height="450px", key="pie1")
                with row1_right:
                    option = {
                        "title": {"text": "Sector Exposure", "left": "center"},
                        "toolbox": {
                            "show": True,
                            "feature": {
                                "mark": {"show": True},
                                "dataView": {"show": True, "readOnly": False},
                                "restore": {"show": True},
                                "saveAsImage": {"show": True},
                            },
                        },
                        "series": [
                            {
                                "name": "Sector Exposure",
                                "type": "pie",
                                "radius": [80, 180],
                                "center": ["50%", "50%"],
                                "roseType": "area",
                                "itemStyle": {"borderRadius": 2},
                                "label": {
                                    "position": "outside",
                                    "alignTo": "labelLine",
                                    "edgeDistance": 2
                                },
                                "labelLine": {
                                    "length": 6,
                                    "length2": 6,
                                    "smooth": True
                                },
                                "labelLayout": {
                                    "hideOverlap": False,
                                    "moveOverlap": "shiftY"
                                },
                                "data": [
                                    {"value": 22.0, "name": "Financials"},
                                    {"value": 14.1, "name": "Technology"},
                                    {"value": 13.0, "name": "Industrials"},
                                    {"value": 8.1, "name": "Energy"},
                                    {"value": 6.4, "name": "Healthcare"},
                                    {"value": 6.0, "name": "Materials"},
                                    {"value": 5.6, "name": "Consumer Discretionary"},
                                    {"value": 4.7, "name": "Consumer Staples"},
                                    {"value": 4.4, "name": "Communication Services"},
                                    {"value": 3.8, "name": "Utilities"},
                                    {"value": 2.6, "name": "Real Estate"},
                                ],
                            }
                        ],
                    }
                    st_echarts(options=option, height="500px")

                row2_left, row2_right = st.columns(2)
                with row2_left:
                    options = {
                        "title": {"text": "Market Capitalization", "left": "center"},
                        "tooltip": {"trigger": "item"},
                        "series": [
                            {
                                "name": "Market Cap Exposure",
                                "type": "pie",
                                "radius": ["40%", "70%"],
                                "center": ["50%", "70%"],
                                "startAngle": 180,
                                "endAngle": 360,
                                "label": {
                                    "formatter": "{b}: {c}%",
                                },
                                "data": [
                                    {"value": 37.0, "name": "Mega Cap"},
                                    {"value": 38.0, "name": "Large Cap"},
                                    {"value": 11.0, "name": "Mid Cap"},
                                    {"value": 14.0, "name": "Small Cap"},

                                ],
                            }
                        ],
                    }
                    st_echarts(options=options, height="500px")
                with row2_right:
                    label_color, grid_color = get_chart_theme_colors()
                    radar_text_color = label_color
                    radar_panel_bg = "rgba(15,23,42,0.95)"
                    radar_panel_text_color = "#f8fafc"
                    radar_split_area_colors = ["rgba(255,255,255,0.08)", "rgba(255,255,255,0.14)"]
                    radar_option = {
                        "backgroundColor": "transparent",
                        "title": {
                            "text": "Portfolio Allocation Comparison",
                            "left": "center",
                            "textStyle": {"color": radar_text_color}
                        },
                        "legend": {
                            "data": ["100/0", "70/30", "50/50"],
                            "top": "bottom",
                            "textStyle": {"color": radar_text_color}
                        },
                        "tooltip": {
                            "trigger": "item",
                            "confine": True,
                            "appendToBody": True,
                            "backgroundColor": radar_panel_bg,
                            "borderColor": radar_panel_bg,
                            "borderWidth": 1,
                            "textStyle": {"color": radar_panel_text_color},
                            "extraCssText": f"box-shadow: 0 0 10px rgba(0,0,0,0.25); color: {radar_panel_text_color};"
                        },
                        "radar": {
                            "center": ["50%", "50%"],
                            "radius": "70%",
                            "axisName": {"textStyle": {"color": radar_text_color, "fontSize": 13}},
                            "axisLine": {"lineStyle": {"color": radar_text_color, "width": 1}},
                            "splitLine": {"lineStyle": {"color": grid_color, "width": 1}},
                            "axisLabel": {"show": True, "textStyle": {"color": radar_text_color, "fontSize": 12}},
                            "splitArea": {"areaStyle": {"color": radar_split_area_colors}},
                            "indicator": [
                                {"name": "US Equities", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "US Small Cap Value", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "International Small Cap Value", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "International Developed", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "Emerging Markets", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "Canada Equities", "max": 50, "textStyle": {"color": radar_text_color}},
                                {"name": "Canadian Bonds", "max": 50, "textStyle": {"color": radar_text_color}},
                            ]
                        },
                        "series": [
                            {
                                "name": "Allocation",
                                "type": "radar",
                                "symbol": "circle",
                                "symbolSize": 8,
                                "lineStyle": {"width": 3},
                                "areaStyle": {"opacity": 0.2},
                                "emphasis": {"focus": "series"},
                                "data": [
                                    {
                                        "value": [25.0, 8.0, 8.0, 20.0, 10.0, 29.0, 0.0],
                                        "name": "100/0",
                                        "itemStyle": {"color": "#ff4d4f"},
                                        "lineStyle": {"color": "#ff4d4f"},
                                        "areaStyle": {"color": "#ff4d4f"},
                                    },
                                    {
                                        "value": [21.0, 7.0, 4.2, 11.2, 5.6, 21.0, 30.0],
                                        "name": "70/30",
                                        "itemStyle": {"color": "#2f54eb"},
                                        "lineStyle": {"color": "#2f54eb"},
                                        "areaStyle": {"color": "#2f54eb"},
                                    },
                                    {
                                        "value": [15.0, 5.0, 3.0, 8.0, 4.0, 15.0, 50.0],
                                        "name": "50/50",
                                        "itemStyle": {"color": "#13c2c2"},
                                        "lineStyle": {"color": "#13c2c2"},
                                        "areaStyle": {"color": "#13c2c2"},
                                    },
                                ],
                            }
                        ],
                    }
                    st_echarts(radar_option, height="500px", theme="dark")
            except Exception:
                # fallback to plotly if echarts fails
                fig1 = px.pie(names=list(pie1.keys()), values=list(pie1.values()), title=pie1_title)
                fig1.update_traces(textposition='inside', textinfo='label+percent')
                row1_left, row1_right = st.columns(2)
                with row1_left:
                    row1_left.plotly_chart(fig1, use_container_width=True)
                with row1_right:
                    st.write("Sector allocation chart unavailable")
                row2_left, row2_right = st.columns(2)
                with row2_left:
                    st.write("Market cap chart unavailable")
                with row2_right:
                    st.write("Portfolio allocation comparison chart unavailable")
        except Exception:
            pass

with references_tab:
    # -----------------------------
    # Bottom: Top Countries by Rank table
    # -----------------------------
    try:
        st.markdown("---")
        st.header("🌍 Top Countries by Rank")

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
             ("Ireland",28.14),("Brazil",27.65),("Canada",27.56)],
        ]

        import colorsys as _cs
        all_countries_sorted = sorted({c for block in blocks for c, _ in block})
        n = len(all_countries_sorted)
        country_colors = {
            c: "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for i, c in enumerate(all_countries_sorted)
            for r, g, b in [_cs.hsv_to_rgb(i / n, 0.58, 0.86)]
        }

        def _fg(hex_bg):
            h = hex_bg.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#111827"

        lookup = {
            year: {rank: (c, v) for rank, (c, v) in enumerate(block, start=1)}
            for year, block in zip(years, blocks)
        }

        # Use explicit palette for reliable contrast in rendered HTML table.
        border = "1px solid rgba(255,255,255,0.18)"
        header_text = "#f1f5f9"
        header_bg = "rgba(255,255,255,0.10)"

        th_style = (
            f"padding:2px 2px;"
            f"color:{header_text};"
            f"background:{header_bg};"
            f"text-align:center;"
            f"border:{border};"
            f"font-size:13px;"
            f"width:60px;"
            f"min-width:60px;"
        )

        rank_th_style = (
            f"padding:4px 4px;"
            f"color:{header_text};"
            f"background:{header_bg};"
            f"text-align:center;"
            f"border:{border};"
            f"font-size:13px;"
            f"width:40px;"
            f"min-width:40px;"
        )

        header_html = f"<tr><th style='{rank_th_style}'>Rank</th>"
        for year in years:
            header_html += f"<th style='{th_style}'>{year}</th>"
        header_html += "</tr>"

        rows_html = ""
        for rank in range(1, 11):
            row = f"<td style='padding:8px 6px;font-weight:bold;text-align:center;color:{header_text};background:{header_bg};border:{border};width:44px;min-width:44px;'>{rank}</td>"
            for year in years:
                entry = lookup.get(year, {}).get(rank)
                if entry:
                    country, pct = entry
                    bg = country_colors[country]
                    fg = _fg(bg)
                    row += (
                        f"<td style='padding:8px 6px;background:{bg};color:{fg};text-align:center;"
                        f"border:1px solid rgba(0,0,0,0.1);width:60px;min-width:60px;line-height:1.5;'>"
                        f"<span style='font-size:14px;font-weight:bold;display:block;'>{country}</span>"
                        f"<span style='font-size:11px;'>{pct:.2f}%</span></td>"
                    )
                else:
                    row += f"<td style='border:{border};width:90px;min-width:90px;'></td>"
            rows_html += f"<tr>{row}</tr>"

        st.markdown(
            "<div style='overflow-x:auto;margin-top:8px;'>"
            "<table style='border-collapse:collapse;width:80%;font-family:Arial,sans-serif;'>"
            f"<thead>{header_html}</thead><tbody>{rows_html}</tbody>"
            "</table></div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.warning("Failed to render Top Countries by Rank table.")


    try:
        st.markdown("---")
        st.header("⭐ Example Morning Star Equity Mix")

        hm_text = "#f1f5f9"
        hm_cell_text = "#0f172a"
        hm_grid = "rgba(255,255,255,0.18)"
        hm_echarts_theme = "dark"

        style_box_opts = {
            "backgroundColor": "transparent",
            "textStyle": {
                "color": hm_text,
                "fontSize": 14,
            },
            "legend": {
                "data": ["Style Box"],
                "orient": "horizontal",
                "bottom": "2%",
                "left": "center",
                "textStyle": {
                    "fontSize": 14,
                    "color": hm_text
            }
            },
            "grid": {"height": "70%", "top": "10%", "left": "12%", "right": "6%"},
            "tooltip": {
                "textStyle": {
                    "fontSize": 14,
                    "color": hm_text
                },
                "backgroundColor": "var(--secondary-background-color)",
            },
            "xAxis": {
                "type": "category",
                "data": ["Value", "Blend", "Growth"],
                "axisLabel": {"fontSize": 14, "color": hm_text},
                "axisLine": {"lineStyle": {"color": hm_text}},
                "splitLine": {"lineStyle": {"color": hm_grid}},
            },
            "yAxis": {
                "type": "category",
                "data": ["Large", "Mid", "Small"],
                "inverse": True,
                "axisLabel": {"fontSize": 14, "color": hm_text},
                "axisLine": {"lineStyle": {"color": hm_text}},
                "splitLine": {"lineStyle": {"color": hm_grid}},
            },
            "visualMap": {
                "type": "piecewise",
                "pieces": [
                    {"min": 0, "max": 10, "label": "0-10%", "color": "#dbeafe"},
                    {"min": 10, "max": 25, "label": "10-25%", "color": "#93c5fd"},
                    {"min": 25, "max": 50, "label": "25-50%", "color": "#60a5fa"},
                    {"min": 50, "label": ">50%", "color": "#3b82f6"},
                ],
                "orient": "horizontal",
                "bottom": 0,
                "textStyle": {"color": hm_text},
            },
            "series": [{
                "type": "heatmap",
                "data": [
                    [0, 0, 35], [1, 0, 10], [2, 0, 5],
                    [0, 1, 15], [1, 1, 20], [2, 1, 0],
                    [0, 2, 5],  [1, 2, 10], [2, 2, 0],
                ],
                "itemStyle": {"borderWidth": 2},
                "label": {
                    "show": True,
                    "formatter": "{@[2]}%",
                    "fontSize": 24,
                    "fontWeight": "bold",
                    "color": hm_cell_text,
                    "textBorderColor": "rgba(241,245,249,0.95)",
                    "textBorderWidth": 2,
                },
            }],
        }

        col1, col2 = st.columns([0.7, 1.3])

        with col1:
            st_echarts(options=style_box_opts, height="400px", theme=hm_echarts_theme)

        with col2:

            st.markdown("""
            #### Size (Market Cap)

            - **Large Cap**: > ~$10B market value  
            Typically established companies with lower volatility and stable earnings.

            - **Mid Cap**: approx. \\$2B - $10B  
            Balanced mix of growth potential and stability.

            - **Small Cap**: < ~$2B  
            Higher growth potential, but more volatile and riskier.

            ---

            #### Style (Investment Type)

            - **Value**: Undervalued relative to fundamentals (price vs earnings/book)  
            - **Growth**: Expected to grow earnings/revenue faster than average  
            - **Blend**: Mix of value and growth characteristics
            """)

    except Exception:
        st.warning("Failed to render Morning Star Equity Mix chart.")