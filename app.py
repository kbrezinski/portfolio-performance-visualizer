# Fix for yfinance cache
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import tempfile

# Configure yfinance cache
yf.set_tz_cache_location(tempfile.gettempdir())

# Set page config
st.set_page_config(
    page_title="Portfolio Performance Visualizer",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        color: #1E88E5;
    }
    .portfolio-editor {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for portfolios if not exists
if 'benchmark_portfolio' not in st.session_state:
    st.session_state.benchmark_portfolio = {
        "XIU.TO": 0.35,  # iShares S&P/TSX 60 Index ETF
        "VFV.TO": 0.65,  # Vanguard S&P500 Index ETF
        "ZAG.TO": 0.00   # BMO Aggregate Bond Index ETF
    }

if 'custom_portfolio' not in st.session_state:
    st.session_state.custom_portfolio = {
        "AVDV": 0.08,    # Avantis International Small Cap Value ETF
        "AVUV": 0.08,    # Avantis U.S. Small Cap Value ETF
        "VUN.TO": 0.25,  # Vanguard U.S.Total Market Index ETF
        "XEC.TO": 0.10,  # BlackRock Canada iShares Core
        "XEF.TO": 0.20,  # BlackRock Canada iShares Core
        "XIC.TO": 0.29   # BlackRock iShares Core S&P/TSX Capped Composite Index ETF
    }

if 'custom_portfolio2' not in st.session_state:
    st.session_state.custom_portfolio2 = {}

if 'custom_portfolio3' not in st.session_state:
    st.session_state.custom_portfolio3 = {}

# Title
st.title("📈 Portfolio Performance Visualizer")

# Time period selection in sidebar
st.sidebar.header("Settings")
time_periods = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(time_periods.keys()),
    index=3  # Default to 1 Year
)

# Function to validate and normalize portfolio weights
def normalize_weights(portfolio):
    total = sum(weight for weight in portfolio.values() if weight is not None)
    if total == 0:
        return portfolio
    return {symbol: (weight/total if weight is not None else 0) for symbol, weight in portfolio.items()}

# Function to create portfolio input fields
def portfolio_input_section(portfolio_name, default_portfolio):
    st.markdown(f"<div class='portfolio-editor'>", unsafe_allow_html=True)
    st.subheader(f"{portfolio_name} Composition")
    
    # Initialize empty portfolio
    portfolio = {}
    total_weight = 0
    
    # Create 10 rows of single column inputs
    for i in range(10):
        col1, col2 = st.columns([3, 1])
        
        # Symbol and weight inputs
        default_symbol = list(default_portfolio.keys())[i] if i < len(default_portfolio) else ""
        default_weight = list(default_portfolio.values())[i] if i < len(default_portfolio) else 0.0
        
        symbol = col1.text_input(
            f"Symbol {i+1}",
            value=default_symbol,
            key=f"{portfolio_name}_symbol_{i}"
        ).strip().upper()
        
        weight = col2.number_input(
            f"Weight {i+1}",
            min_value=0.0,
            max_value=100.0,
            value=float(default_weight * 100),
            key=f"{portfolio_name}_weight_{i}",
            format="%.1f"
        )
        
        if symbol and weight > 0:
            portfolio[symbol] = weight / 100
            total_weight += weight

    # Show total weight
    st.markdown(f"**Total Weight: {total_weight:.1f}%**")
    if total_weight != 100:
        st.warning("Total weight should be 100%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Normalize weights
    return normalize_weights(portfolio)

# Function to safely calculate returns
def safe_calculate_returns(df):
    if df is None or df.empty:
        return None
    try:
        # Calculate daily returns
        returns = df['Close'].pct_change()
        # Drop NaN values
        returns = returns.dropna()
        if len(returns) == 0:
            return None
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        return cumulative_returns
    except Exception as e:
        st.error(f"Error calculating returns: {str(e)}")
        return None

# Function to fetch and process portfolio data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_portfolio_data(portfolio, start_date, end_date):
    """Fetch and process portfolio data"""
    if not portfolio:  # If portfolio is empty
        return None
        
    try:
        # Initialize DataFrame for storing daily returns
        portfolio_data = pd.DataFrame()
        valid_symbols = []
        
        # Fetch data for each symbol
        for symbol, weight in portfolio.items():
            if weight > 0:  # Only fetch data for symbols with positive weights
                try:
                    # Use download instead of Ticker for more reliable data fetching
                    hist = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False
                    )
                    
                    if not hist.empty and 'Close' in hist.columns:
                        # Calculate daily returns
                        returns = hist['Close'].pct_change()
                        portfolio_data[symbol] = returns
                        valid_symbols.append(symbol)
                    else:
                        st.warning(f"No data available for {symbol}")
                except Exception as e:
                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
        
        if portfolio_data.empty:
            return None
            
        # Drop first row (NaN from pct_change) and any other NaN values
        portfolio_data = portfolio_data.dropna()
        
        if portfolio_data.empty:
            return None
            
        # Recalculate weights for valid symbols only
        total_weight = sum(portfolio[symbol] for symbol in valid_symbols)
        if total_weight == 0:
            return None
            
        # Calculate weighted returns
        weighted_returns = pd.Series(0.0, index=portfolio_data.index)
        for symbol in valid_symbols:
            weight = portfolio[symbol] / total_weight  # Normalize weights
            weighted_returns += portfolio_data[symbol] * weight
        
        # Calculate cumulative returns
        cumulative_returns = (1 + weighted_returns).cumprod()
        
        return cumulative_returns
    except Exception as e:
        st.error(f"Error processing portfolio data: {str(e)}")
        return None

# Main content
try:
    # Get dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_periods[selected_period])
    
    # Show loading state
    with st.spinner("Loading portfolio data..."):
        # Get portfolio returns
        benchmark_returns = get_portfolio_data(st.session_state.benchmark_portfolio, start_date, end_date)
        custom_returns = get_portfolio_data(st.session_state.custom_portfolio, start_date, end_date)
        custom_returns2 = get_portfolio_data(st.session_state.custom_portfolio2, start_date, end_date)
        custom_returns3 = get_portfolio_data(st.session_state.custom_portfolio3, start_date, end_date)
        
        # Create comparison plot if we have any valid data
        if benchmark_returns is not None or custom_returns is not None or \
           custom_returns2 is not None or custom_returns3 is not None:
            
            fig = go.Figure()
            
            # Add portfolio lines only if they have valid data
            if benchmark_returns is not None:
                fig.add_trace(go.Scatter(
                    x=benchmark_returns.index,
                    y=benchmark_returns,
                    name='Benchmark Portfolio',
                    line=dict(color='#1f77b4')
                ))
            
            if custom_returns is not None:
                fig.add_trace(go.Scatter(
                    x=custom_returns.index,
                    y=custom_returns,
                    name='Custom Portfolio 1',
                    line=dict(color='#ff7f0e')
                ))
            
            if custom_returns2 is not None and not custom_returns2.empty:
                fig.add_trace(go.Scatter(
                    x=custom_returns2.index,
                    y=custom_returns2,
                    name='Custom Portfolio 2',
                    line=dict(color='#2ca02c')
                ))
            
            if custom_returns3 is not None and not custom_returns3.empty:
                fig.add_trace(go.Scatter(
                    x=custom_returns3.index,
                    y=custom_returns3,
                    name='Custom Portfolio 3',
                    line=dict(color='#d62728')
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Portfolio Performance Comparison ({selected_period})",
                yaxis_title="Cumulative Return (1 = Initial Investment)",
                xaxis_title="Date",
                template="plotly_white",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display performance metrics
            st.subheader("Performance Metrics")
            metrics_cols = st.columns(4)
            
            # Helper function to safely display metrics
            def display_metric(returns, name, col, benchmark_return=None):
                if returns is not None and len(returns) > 0:
                    total_return = (returns.iloc[-1] - 1) * 100
                    delta = None if benchmark_return is None else f"{(total_return - benchmark_return):.2f}%"
                    col.metric(name, f"{total_return:.2f}%", delta)
                else:
                    col.metric(name, "No data", None)
            
            # Display metrics for each portfolio
            benchmark_return = None
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                benchmark_return = (benchmark_returns.iloc[-1] - 1) * 100
            
            display_metric(benchmark_returns, "Benchmark Portfolio", metrics_cols[0])
            display_metric(custom_returns, "Custom Portfolio 1", metrics_cols[1], benchmark_return)
            display_metric(custom_returns2, "Custom Portfolio 2", metrics_cols[2], benchmark_return)
            display_metric(custom_returns3, "Custom Portfolio 3", metrics_cols[3], benchmark_return)
        
        else:
            st.warning("No valid data available for any portfolio. Please check your portfolio compositions and try again.")
        
        # Portfolio editors
        st.header("Portfolio Management")
        
        # Create four columns for portfolio editors
        portfolio_cols = st.columns(4)
        
        # Portfolio input sections in columns
        with portfolio_cols[0]:
            st.session_state.benchmark_portfolio = portfolio_input_section(
                "Benchmark Portfolio",
                st.session_state.benchmark_portfolio
            )

        with portfolio_cols[1]:
            st.session_state.custom_portfolio = portfolio_input_section(
                "Custom Portfolio 1",
                st.session_state.custom_portfolio
            )
        
        with portfolio_cols[2]:
            st.session_state.custom_portfolio2 = portfolio_input_section(
                "Custom Portfolio 2",
                st.session_state.custom_portfolio2
            )
        
        with portfolio_cols[3]:
            st.session_state.custom_portfolio3 = portfolio_input_section(
                "Custom Portfolio 3",
                st.session_state.custom_portfolio3
            )

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.error("Stack trace:", exc_info=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Data provided by Yahoo Finance</p>
    </div>
""", unsafe_allow_html=True) 