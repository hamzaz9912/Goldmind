import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_fetch import DataFetcher
from indicators import TechnicalIndicators
from model_inference import ModelInference
from utils import *

# Page configuration
st.set_page_config(
    page_title="AI Scalping Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Main title
st.title("🤖 AI-Driven Scalping Dashboard")
st.markdown("Real-time Buy/Sell signals for Gold & Top 20 Cryptocurrencies")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Market type selection
market_type = st.sidebar.selectbox(
    "Market Type",
    ["Gold", "Crypto"],
    key="market_type"
)

# Asset selection based on market type
if market_type == "Gold":
    asset = "XAUUSD=X"
    asset_display = "Gold (XAUUSD)"
else:
    crypto_assets = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum", 
        "BNB-USD": "Binance Coin",
        "SOL-USD": "Solana",
        "XRP-USD": "Ripple",
        "ADA-USD": "Cardano",
        "AVAX-USD": "Avalanche",
        "DOT-USD": "Polkadot",
        "MATIC-USD": "Polygon",
        "LINK-USD": "Chainlink",
        "UNI-USD": "Uniswap",
        "LTC-USD": "Litecoin",
        "BCH-USD": "Bitcoin Cash",
        "ALGO-USD": "Algorand",
        "VET-USD": "VeChain",
        "ICP-USD": "Internet Computer",
        "ATOM-USD": "Cosmos",
        "FIL-USD": "Filecoin",
        "TRX-USD": "TRON",
        "ETC-USD": "Ethereum Classic"
    }
    
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        list(crypto_assets.keys()),
        format_func=lambda x: crypto_assets[x],
        key="crypto_asset"
    )
    asset = selected_crypto
    asset_display = crypto_assets[selected_crypto]

# Timeframe selection
timeframes = {
    "5m": 5,
    "15m": 15, 
    "30m": 30,
    "1h": 60,
    "1d": 1440
}

timeframe = st.sidebar.selectbox(
    "Timeframe",
    list(timeframes.keys()),
    index=2,  # Default to 30m
    key="timeframe"
)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=st.session_state.auto_refresh,
    key="auto_refresh_toggle"
)

if auto_refresh != st.session_state.auto_refresh:
    st.session_state.auto_refresh = auto_refresh

# Display refresh info  
refresh_interval = timeframes[timeframe] * 60  # Convert to seconds
time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()

if auto_refresh:
    st.sidebar.info(f"🔄 Auto-refresh every {timeframe}")
    
    # Check if it's time to refresh
    if time_since_refresh >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Manual refresh button
if st.sidebar.button("🔄 Manual Refresh"):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Helper functions
def create_candlestick_chart(df, title):
    """Create an interactive candlestick chart with technical indicators"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{title} Price', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add EMAs
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_20'],
                name='EMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_50'],
                name='EMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add VWAP for crypto
    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['VWAP'],
                name='VWAP',
                line=dict(color='purple', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # RSI subplot
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI_14'],
                name='RSI',
                line=dict(color='yellow', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD subplot
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        if 'MACDs_12_26_9' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACDs_12_26_9'],
                    name='Signal',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
        
        if 'MACDh_12_26_9' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACDh_12_26_9'],
                    name='Histogram',
                    marker_color='gray',
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'{title} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_line_chart(df, column, title):
    """Create a simple line chart"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column] if column in df.columns else df.iloc[:, 0],
            name=title,
            line=dict(color='orange', width=2)
        )
    )
    
    fig.update_layout(
        title=title,
        height=200,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig

def display_signal_panel(signal_data, latest_candle):
    """Display the trading signal panel"""
    st.subheader("🎯 Trading Signal")
    
    if signal_data:
        signal = signal_data.get('signal', 'NO TRADE')
        confidence = signal_data.get('confidence', 0)
        entry = signal_data.get('entry', latest_candle['Close'])
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)
        
        # Signal color coding
        if signal == 'BUY':
            signal_color = '🟢'
            signal_bg = 'success'
        elif signal == 'SELL':
            signal_color = '🔴'
            signal_bg = 'error'
        else:
            signal_color = '⚪'
            signal_bg = 'secondary'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signal", f"{signal_color} {signal}")
            st.metric("Confidence", f"{confidence:.1f}%")
            
        with col2:
            st.metric("Entry", f"{entry:.4f}")
            st.metric("Stop Loss", f"{stop_loss:.4f}")
            
        with col3:
            st.metric("Take Profit", f"{take_profit:.4f}")
            risk_reward = abs(take_profit - entry) / abs(entry - stop_loss) if stop_loss != entry else 0
            st.metric("Risk:Reward", f"1:{risk_reward:.2f}")
            
    else:
        st.info("⏳ Generating signal...")

# Initialize data fetcher and other components
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data(asset, timeframe, periods=200):
    fetcher = DataFetcher()
    return fetcher.get_ohlcv_data(asset, timeframe, periods)

@st.cache_data(ttl=300)
def calculate_indicators(df):
    indicators = TechnicalIndicators()
    return indicators.calculate_all_indicators(df)

@st.cache_data(ttl=300)
def get_fundamentals_data(market_type):
    fetcher = DataFetcher()
    if market_type == "Gold":
        return fetcher.get_dxy_data(), fetcher.get_forex_news()
    else:
        return fetcher.get_btc_dominance(), fetcher.get_crypto_news()

# Main dashboard layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 {asset_display} - {timeframe}")
    
    # Fetch and process data
    try:
        with st.spinner("Fetching market data..."):
            df = fetch_market_data(asset, timeframe)
            
        if df is not None and not df.empty:
            # Calculate technical indicators
            df_with_indicators = calculate_indicators(df)
            
            # Create the candlestick chart
            fig = create_candlestick_chart(df_with_indicators, asset_display)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate ML signal
            model_inference = ModelInference()
            signal_data = model_inference.generate_signal(df_with_indicators, asset)
            
            # Display signal information
            display_signal_panel(signal_data, df_with_indicators.iloc[-1])
            
        else:
            st.error("❌ Unable to fetch market data. Please try again.")
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

with col2:
    st.subheader("📈 Fundamentals")
    
    try:
        fund_data1, fund_data2 = get_fundamentals_data(market_type)
        
        if market_type == "Gold":
            # Display DXY chart
            st.write("**💵 Dollar Index (DXY)**")
            if fund_data1 is not None and not fund_data1.empty:
                dxy_fig = create_line_chart(fund_data1, "DXY", "Dollar Index")
                st.plotly_chart(dxy_fig, use_container_width=True)
            else:
                st.info("DXY data unavailable")
                
            # Display forex news
            st.write("**📰 Economic Events**")
            if fund_data2:
                for i, news in enumerate(fund_data2[:5]):
                    st.write(f"• {news}")
            else:
                st.info("No recent economic events")
                
        else:
            # Display BTC dominance
            st.write("**₿ BTC Dominance**")
            if fund_data1 is not None:
                st.metric("BTC Dominance", f"{fund_data1:.2f}%")
            else:
                st.info("BTC dominance unavailable")
                
            # Display crypto news
            st.write("**📰 Crypto News**")
            if fund_data2:
                for i, news in enumerate(fund_data2[:5]):
                    st.write(f"• {news}")
            else:
                st.info("No recent crypto news")
                
    except Exception as e:
        st.error(f"Error loading fundamentals: {str(e)}")

# Technical indicators panel
st.subheader("📊 Technical Analysis")

if 'df_with_indicators' in locals() and df_with_indicators is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df_with_indicators.iloc[-1]
    
    with col1:
        st.metric(
            "RSI(14)", 
            f"{latest.get('RSI_14', 0):.2f}",
            delta=None
        )
        
    with col2:
        ema20 = latest.get('EMA_20', 0)
        ema50 = latest.get('EMA_50', 0)
        trend = "🟢 Bullish" if ema20 > ema50 else "🔴 Bearish"
        st.metric("EMA Trend", trend)
        
    with col3:
        macd = latest.get('MACD_12_26_9', 0)
        macd_signal = latest.get('MACDs_12_26_9', 0)
        macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
        st.metric("MACD", macd_trend)
        
    with col4:
        atr = latest.get('ATR_14', 0)
        st.metric("ATR(14)", f"{atr:.4f}")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Disclaimer**: This is for educational purposes only. "
    "Always do your own research before making trading decisions."
)

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(1)  # Small delay to prevent excessive CPU usage
    placeholder = st.empty()
    with placeholder.container():
        remaining_time = refresh_interval - time_since_refresh
        if remaining_time > 0:
            st.info(f"⏱️ Next refresh in {int(remaining_time)} seconds")
    
    # Create subplots
    fig = make_subplots(
