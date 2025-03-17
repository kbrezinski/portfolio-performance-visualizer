import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

# Configure page
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 2.5rem !important;
        color: #1E88E5;
    }
    .portfolio-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'benchmark_portfolio' not in st.session_state:
    st.session_state.benchmark_portfolio = {
        "XIU.TO": 0.35,
        "VFV.TO": 0.65
    }

if 'custom_portfolio' not in st.session_state:
    st.session_state.custom_portfolio = {
        "VUN.TO": 0.40,
        "XEF.TO": 0.30,
        "XIC.TO": 0.30
    }

# Title
st.title("📈 Portfolio Visualizer")

# Sidebar settings
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

def fetch_stock_data(symbol, start_date, end_date, retries=3):
    """
    Fetch stock data with retries and proper error handling
    """
    for attempt in range(retries):
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if not data.empty and 'Close' in data.columns:
                return data
            
            time.sleep(1)  # Wait before retry
            
        except Exception as e:
            if attempt == retries - 1:
                st.warning(f"Failed to fetch data for {symbol} after {retries} attempts: {str(e)}")
                return None
            time.sleep(1)  # Wait before retry
    
    return None

@st.cache_data(ttl=300)
def get_portfolio_data(portfolio, start_date, end_date):
    """
    Fetch and process portfolio data with improved error handling
    """
    if not portfolio:
        return None
    
    try:
        all_data = {}
        valid_symbols = []
        
        # Fetch data for each symbol
        for symbol, weight in portfolio.items():
            if weight > 0:
                data = fetch_stock_data(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    valid_symbols.append(symbol)
                    st.info(f"Successfully fetched data for {symbol}")
        
        if not valid_symbols:
            st.warning("No valid data was fetched for any symbols")
            return None
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(dtype=float)
        total_weight = sum(portfolio[symbol] for symbol in valid_symbols)
        
        for symbol in valid_symbols:
            weight = portfolio[symbol] / total_weight
            returns = all_data[symbol]['Close'].pct_change()
            
            if portfolio_returns.empty:
                portfolio_returns = returns * weight
            else:
                portfolio_returns = portfolio_returns.add(returns * weight, fill_value=0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        return cumulative_returns
    
    except Exception as e:
        st.error(f"Error processing portfolio data: {str(e)}")
        return None

def portfolio_editor(portfolio_name, default_portfolio):
    """
    Create portfolio input section
    """
    st.markdown(f"<div class='portfolio-section'>", unsafe_allow_html=True)
    st.subheader(f"{portfolio_name}")
    
    portfolio = {}
    total_weight = 0
    
    # Create input fields
    for i in range(5):  # Limit to 5 symbols for simplicity
        col1, col2 = st.columns([3, 1])
        
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
    
    st.markdown(f"**Total Weight: {total_weight:.1f}%**")
    if abs(total_weight - 100) > 0.1:
        st.warning("Total weight should be 100%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    return {k: v/total_weight*100 if total_weight > 0 else v for k, v in portfolio.items()}

# Main content
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_periods[selected_period])
    
    # Portfolio editors
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.benchmark_portfolio = portfolio_editor(
            "Benchmark Portfolio",
            st.session_state.benchmark_portfolio
        )
    
    with col2:
        st.session_state.custom_portfolio = portfolio_editor(
            "Custom Portfolio",
            st.session_state.custom_portfolio
        )
    
    # Fetch and display data
    with st.spinner("Loading portfolio data..."):
        benchmark_returns = get_portfolio_data(st.session_state.benchmark_portfolio, start_date, end_date)
        custom_returns = get_portfolio_data(st.session_state.custom_portfolio, start_date, end_date)
        
        if benchmark_returns is not None or custom_returns is not None:
            fig = go.Figure()
            
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
                    name='Custom Portfolio',
                    line=dict(color='#ff7f0e')
                ))
            
            fig.update_layout(
                title=f"Portfolio Performance Comparison ({selected_period})",
                yaxis_title="Cumulative Return",
                xaxis_title="Date",
                template="plotly_white",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            def display_metrics(returns, name, col):
                if returns is not None and len(returns) > 0:
                    try:
                        total_return = float(returns.iloc[-1] - 1) * 100
                        col.metric(name, f"{total_return:,.2f}%")
                    except Exception as e:
                        col.metric(name, "Error calculating")
                        st.warning(f"Error calculating {name} metrics: {str(e)}")
                else:
                    col.metric(name, "No data")
            
            display_metrics(benchmark_returns, "Benchmark Portfolio", col1)
            display_metrics(custom_returns, "Custom Portfolio", col2)
        
        else:
            st.warning("No valid data available. Please check your portfolio compositions and try again.")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Data provided by Yahoo Finance</p>
    </div>
""", unsafe_allow_html=True) 