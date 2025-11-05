import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from sklearn.linear_model import LinearRegression
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

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffa500;
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #00ff88;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00ff88, #009944);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #666666, #999999);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .sidebar-content {
        background: linear-gradient(180deg, #1a1a1a, #2d2d2d);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'model_type' not in st.session_state:
    st.session_state.model_type = "RL"
if 'forecast_accuracy_tracking' not in st.session_state:
    st.session_state.forecast_accuracy_tracking = {}

# Main title
st.title("🤖 AI-Driven Scalping Dashboard")
st.markdown("Real-time Buy/Sell signals for Gold & Top 20 Cryptocurrencies")

# Define asset lists for live prices
gold_assets = ['GC=F']
crypto_assets = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD']
forex_assets = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X']

# Fetch live prices for all assets
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_all_live_prices():
    fetcher = DataFetcher()
    all_assets = gold_assets + crypto_assets + forex_assets
    return fetcher.get_live_prices_for_assets(all_assets)

# Fetch live prices globally
live_prices = {}
try:
    live_prices = get_all_live_prices()
except Exception as e:
    st.warning(f"Unable to fetch live prices: {str(e)}")
    live_prices = {}

# Live Prices Section - Prominently displayed at the top
st.header("💰 Live Market Prices")

try:

    # Display live prices in organized sections
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🥇 Gold")
        if 'GC=F' in live_prices:
            st.metric("Gold (GC=F)", f"${live_prices['GC=F']:.2f}")
        else:
            st.metric("Gold (GC=F)", "N/A")

    with col2:
        st.subheader("₿ Top Cryptocurrencies")
        for asset in crypto_assets[:4]:  # Show first 4 cryptos
            if asset in live_prices:
                asset_name = asset.replace('-USD', '')
                st.metric(f"{asset_name}", f"${live_prices[asset]:.2f}")
            else:
                asset_name = asset.replace('-USD', '')
                st.metric(f"{asset_name}", "N/A")

    with col3:
        st.subheader("💱 Major Forex Pairs")
        for asset in forex_assets[:4]:  # Show first 4 forex pairs
            if asset in live_prices:
                pair_name = asset.replace('=X', '').replace('USD', '/USD')
                st.metric(f"{pair_name}", f"{live_prices[asset]:.4f}")
            else:
                pair_name = asset.replace('=X', '').replace('USD', '/USD')
                st.metric(f"{pair_name}", "N/A")

    # Additional crypto assets in a second row if needed
    if len(crypto_assets) > 4:
        st.subheader("₿ Additional Cryptocurrencies")
        cols = st.columns(4)
        for i, asset in enumerate(crypto_assets[4:]):
            with cols[i % 4]:
                if asset in live_prices:
                    asset_name = asset.replace('-USD', '')
                    st.metric(f"{asset_name}", f"${live_prices[asset]:.2f}")
                else:
                    asset_name = asset.replace('-USD', '')
                    st.metric(f"{asset_name}", "N/A")

    # Additional forex pairs
    if len(forex_assets) > 4:
        st.subheader("💱 Additional Forex Pairs")
        cols = st.columns(len(forex_assets) - 4)
        for i, asset in enumerate(forex_assets[4:]):
            with cols[i]:
                if asset in live_prices:
                    pair_name = asset.replace('=X', '').replace('USD', '/USD')
                    st.metric(f"{pair_name}", f"{live_prices[asset]:.4f}")
                else:
                    pair_name = asset.replace('=X', '').replace('USD', '/USD')
                    st.metric(f"{pair_name}", "N/A")

except Exception as e:
    st.error(f"Error fetching live prices: {str(e)}")

st.markdown("---")

# Model type selection
model_types = {
    "RL (Reinforcement Learning)": "RL",
    "LSTM (Long Short-Term Memory)": "LSTM",
    "AutoML (Automated ML)": "AutoML",
    "ML (Traditional Machine Learning)": "ML"
}

selected_model_display = st.sidebar.selectbox(
    "Model Type",
    list(model_types.keys()),
    index=list(model_types.values()).index(st.session_state.model_type) if st.session_state.model_type in model_types.values() else 0,
    key="model_type_select"
)

if model_types[selected_model_display] != st.session_state.model_type:
    st.session_state.model_type = model_types[selected_model_display]

# Current model status
col1, col2 = st.columns([3, 1])
with col2:
    st.info(f"**Model:** {selected_model_display}")
    st.success(f"{selected_model_display} Active")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Market type selection
market_type = st.sidebar.selectbox(
    "Market Type",
    ["Gold", "Crypto", "Forex"],
    key="market_type"
)

# Asset selection based on market type
if market_type == "Gold":
    asset = "GC=F"
    asset_display = "Gold (GC=F)"
elif market_type == "Crypto":
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
else:  # Forex
    forex_assets = {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD",
        "NZDUSD=X": "NZD/USD",
        "EURJPY=X": "EUR/JPY",
        "GBPJPY=X": "GBP/JPY",
        "EURGBP=X": "EUR/GBP"
    }

    selected_forex = st.sidebar.selectbox(
        "Select Forex Pair",
        list(forex_assets.keys()),
        format_func=lambda x: forex_assets[x],
        key="forex_asset"
    )
    asset = selected_forex
    asset_display = forex_assets[selected_forex]

# Timeframe selection
timeframes = {
    "5m": 5,
    "10m": 10
}

timeframe = st.sidebar.selectbox(
    "Timeframe",
    list(timeframes.keys()),
    index=0,  # Default to 5m
    key="timeframe"
)

# Forecast duration selection
forecast_durations = {
    "5 minutes": 5,
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
    "1 hour": 60,
    "2 hours": 120,
    "4 hours": 240
}

forecast_duration = st.sidebar.selectbox(
    "Forecast Duration",
    list(forecast_durations.keys()),
    index=0,  # Default to 5 minutes
    key="forecast_duration"
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
    """Create a professional candlestick chart with enhanced technical indicators"""

    # Create subplots with better proportions
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'📊 {title} Price Action', '📈 RSI Momentum', '🎯 MACD Trend', '💹 Volume Flow'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # Enhanced candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )

    # Add EMAs with enhanced styling
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_20'],
                name='EMA 20',
                line=dict(color='#ffaa00', width=2, dash='solid'),
                mode='lines'
            ),
            row=1, col=1
        )

    if 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_50'],
                name='EMA 50',
                line=dict(color='#0088ff', width=2, dash='solid'),
                mode='lines'
            ),
            row=1, col=1
        )

    # Add VWAP for crypto with better styling
    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['VWAP'],
                name='VWAP',
                line=dict(color='#aa00ff', width=2, dash='dot'),
                mode='lines'
            ),
            row=1, col=1
        )

    # Enhanced RSI subplot
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI_14'],
                name='RSI',
                line=dict(color='#ffff00', width=3),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255,255,0,0.1)'
            ),
            row=2, col=1
        )

        # RSI levels with better styling
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", line_width=2, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#44ff44", line_width=2, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#888888", line_width=1, row=2, col=1)

        # Add RSI overbought/oversold zones
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="rgba(255,68,68,0.1)", row=2, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="rgba(68,255,68,0.1)", row=2, col=1)

    # Enhanced MACD subplot
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='#0088ff', width=3),
                mode='lines'
            ),
            row=3, col=1
        )

        if 'MACDs_12_26_9' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACDs_12_26_9'],
                    name='Signal',
                    line=dict(color='#ff4444', width=2, dash='dot'),
                    mode='lines'
                ),
                row=3, col=1
            )

        if 'MACDh_12_26_9' in df.columns:
            colors = ['#44ff44' if val >= 0 else '#ff4444' for val in df['MACDh_12_26_9']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACDh_12_26_9'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )

    # Volume subplot
    if 'Volume' in df.columns:
        colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4444'
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=4, col=1
        )

    # Professional layout
    fig.update_layout(
        title={
            'text': f'🎯 {title} - Professional Technical Analysis Dashboard',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#00ff88')
        },
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0.9)',
        paper_bgcolor='rgba(0,0,0,0.95)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        )
    )

    # Enhanced y-axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font=dict(color='#00ff88'))
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], title_font=dict(color='#ffff00'))
    fig.update_yaxes(title_text="MACD", row=3, col=1, title_font=dict(color='#0088ff'))
    fig.update_yaxes(title_text="Volume", row=4, col=1, title_font=dict(color='#aa00ff'))

    # Enhanced x-axis formatting
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        tickformat='%H:%M:%S'
    )

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

def forecast_next_period(df, minutes, interval_seconds=0.30):
    """
    Forecast the next period using advanced forecasting with micro-second precision

    Args:
        df: DataFrame with historical data
        minutes: Number of minutes to forecast
        interval_seconds: Interval between forecast points (default 0.30 seconds)

    Returns:
        List of forecasted prices.
    """
    try:
        steps = int(minutes * 60 / interval_seconds)
        forecasts = []
        temp_df = df.copy()

        # Use multiple forecasting methods for better accuracy
        for i in range(steps):
            if len(temp_df) < 10:
                forecast = temp_df['Close'].iloc[-1]
            else:
                # Advanced forecasting using trend analysis and momentum
                closes = temp_df['Close'].tail(30).values  # Use more data points

                # Calculate trend using linear regression
                X = np.arange(len(closes)).reshape(-1, 1)
                y = closes
                model = LinearRegression()
                model.fit(X, y)

                # Get trend slope and intercept
                slope = model.coef_[0]
                intercept = model.intercept_

                # Forecast next step with momentum consideration
                next_x = len(closes)
                base_forecast = model.predict(np.array([[next_x]]))[0]

                # Add momentum factor based on recent price changes
                recent_changes = np.diff(closes[-5:])  # Last 5 changes
                momentum = np.mean(recent_changes) if len(recent_changes) > 0 else 0

                # Combine trend and momentum
                forecast = base_forecast + (momentum * 0.3)  # 30% weight to momentum

                # Add some micro-volatility based on ATR if available
                if 'ATR_14' in temp_df.columns and len(temp_df) > 0:
                    atr = temp_df['ATR_14'].iloc[-1]
                    # Add small random component based on ATR
                    volatility_factor = atr * 0.001  # Very small volatility
                    forecast += np.random.normal(0, volatility_factor)

                # Ensure forecast stays within reasonable bounds
                last_close = closes[-1]
                max_change = last_close * 0.005  # Max 0.5% change per 0.3 second step
                forecast = np.clip(forecast, last_close - max_change, last_close + max_change)

            forecasts.append(forecast)

            # Append forecast to temp_df for next iteration
            new_row = temp_df.iloc[-1].copy()
            new_row.name = temp_df.index[-1] + pd.Timedelta(seconds=interval_seconds)
            new_row['Close'] = forecast
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

        return forecasts

    except Exception as e:
        print(f"Error forecasting: {str(e)}")
        return [df['Close'].iloc[-1]] * steps

def create_forecast_chart(df, asset_display, forecasts, minutes, interval_seconds=0.30, model_type="RL", signal_data=None):
    """Create a high-precision forecast chart with configurable intervals and trading levels"""
    fig = go.Figure()

    # Historical data - show last 50 points for better context
    historical_data = df.tail(50)
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Historical Price',
            line=dict(color='#00ff88', width=3),
            mode='lines'
        )
    )

    # Forecast points with micro-second precision
    steps = len(forecasts)
    forecast_times = [df.index[-1] + pd.Timedelta(seconds=(i+1)*interval_seconds) for i in range(steps)]

    # Create smooth forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecasts,
            name=f'{model_type} Forecast',
            mode='lines',
            line=dict(color='#ff4444', width=3, shape='spline', smoothing=1.3),
            fill='tonexty',
            fillcolor='rgba(255, 68, 68, 0.1)'
        )
    )

    # Add forecast markers at key intervals (every 10 seconds)
    marker_indices = [i for i in range(0, steps, int(10/interval_seconds)) if i < steps]
    if marker_indices:
        fig.add_trace(
            go.Scatter(
                x=[forecast_times[i] for i in marker_indices],
                y=[forecasts[i] for i in marker_indices],
                name='Forecast Points',
                mode='markers',
                marker=dict(
                    color='#ffaa00',
                    size=6,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                showlegend=False
            )
        )

    # Add trading levels if signal data is available
    if signal_data:
        current_price = df['Close'].iloc[-1]
        entry_price = signal_data.get('entry', current_price)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)

        # Entry level
        fig.add_hline(y=entry_price, line_dash="solid", line_color="#00ff88", line_width=3,
                     annotation_text=f"Entry: {entry_price:.4f}", annotation_position="top right")

        # Stop Loss level
        if stop_loss > 0:
            fig.add_hline(y=stop_loss, line_dash="dash", line_color="#ff4444", line_width=3,
                         annotation_text=f"Stop Loss: {stop_loss:.4f}", annotation_position="bottom right")

        # Take Profit level
        if take_profit > 0:
            fig.add_hline(y=take_profit, line_dash="dot", line_color="#44ff44", line_width=3,
                         annotation_text=f"Take Profit: {take_profit:.4f}", annotation_position="top left")

        # Add risk/reward ratio annotation
        if stop_loss > 0 and take_profit > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0

            fig.add_annotation(
                x=forecast_times[-1],
                y=max(forecasts),
                text=f"Risk:Reward = 1:{rr_ratio:.2f}",
                showarrow=False,
                font=dict(color="#ffaa00", size=14, weight="bold"),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor="#ffaa00",
                borderwidth=2,
                borderpad=4
            )

    # Enhanced layout with better styling
    fig.update_layout(
        title={
            'text': f'🔮 {asset_display} - {model_type} {minutes} Minute Forecast ({interval_seconds}s intervals)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='#00ff88')
        },
        xaxis_title='Time (Real-time)',
        yaxis_title='Price',
        height=500,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.9)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)'
        )
    )

    # Add current price annotation
    current_price = df['Close'].iloc[-1]
    fig.add_annotation(
        x=df.index[-1],
        y=current_price,
        text=f"Current: {current_price:.4f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00ff88",
        ax=50,
        ay=-30,
        font=dict(color="#00ff88", size=12),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor="#00ff88",
        borderwidth=1
    )

    return fig

def create_multi_model_comparison_chart(df, asset_display, model_forecasts, minutes):
    """Create a professional chart comparing forecasts from different models with configurable precision"""
    fig = go.Figure()

    # Historical data - show last 50 points for better context
    historical_data = df.tail(50)
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='📈 Historical Price',
            line=dict(color='#00ff88', width=4),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.1)'
        )
    )

    # Enhanced colors and styles for different models
    model_styles = {
        'RL': {
            'color': '#ff4444',
            'name': '🤖 RL Model',
            'dash': 'solid',
            'width': 4
        },
        'LSTM': {
            'color': '#44ff44',
            'name': '🧠 LSTM Model',
            'dash': 'dot',
            'width': 4
        },
        'AutoML': {
            'color': '#ffaa00',
            'name': '⚡ AutoML Model',
            'dash': 'dash',
            'width': 4
        },
        'ML': {
            'color': '#aa44ff',
            'name': '🎯 ML Model',
            'dash': 'longdash',
            'width': 4
        }
    }

    # Add forecasts from each model with enhanced styling
    for model_type, forecasts in model_forecasts.items():
        if forecasts and len(forecasts) > 0:
            steps = len(forecasts)
            forecast_times = [df.index[-1] + pd.Timedelta(seconds=(i+1)*0.30) for i in range(steps)] # Use 0.30-second interval

            style = model_styles.get(model_type, {
                'color': '#888888',
                'name': f'{model_type} Model',
                'dash': 'solid',
                'width': 3
            })

            # Create filled area for forecast confidence
            upper_bound = [f * 1.002 for f in forecasts]  # 0.2% upper bound
            lower_bound = [f * 0.998 for f in forecasts]  # 0.2% lower bound

            fig.add_trace(
                go.Scatter(
                    x=forecast_times + forecast_times[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor=f'rgba({int(style["color"][1:3], 16)}, {int(style["color"][3:5], 16)}, {int(style["color"][5:7], 16)}, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_type} Confidence',
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=forecasts,
                    name=style['name'],
                    mode='lines',
                    line=dict(
                        color=style['color'],
                        width=style['width'],
                        dash=style['dash'],
                        shape='spline',
                        smoothing=1.3
                    )
                )
            )

            # Add end point markers for each forecast
            if steps > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[forecast_times[-1]],
                        y=[forecasts[-1]],
                        name=f'{model_type} Final',
                        mode='markers',
                        marker=dict(
                            color=style['color'],
                            size=10,
                            symbol='diamond',
                            line=dict(color='white', width=2)
                        ),
                        showlegend=False
                    )
                )

    # Professional layout with enhanced styling
    fig.update_layout(
        title={
            'text': f'🔬 {asset_display} - Multi-Model {minutes} Minute Forecast (0.30s precision)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#00ff88', weight='bold')
        },
        xaxis_title='Time (Real-time with 0.30s intervals)',
        yaxis_title='Price',
        height=600,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0.95)',
        paper_bgcolor='rgba(0,0,0,0.98)',
        font=dict(color='white', size=12),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)',
            tickformat='%H:%M:%S'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)',
            tickformat='.4f'
        )
    )

    # Add range selector for better time navigation
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30s", step="second", stepmode="backward"),
                dict(count=1, label="1m", step="minute", stepmode="backward"),
                dict(count=5, label="5m", step="minute", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(0,0,0,0.7)',
            activecolor='#00ff88'
        )
    )

    return fig

def analyze_dxy_correlation(asset_df, dxy_df):
    """Analyze correlation between asset and DXY"""
    try:
        # Align timeframes (assuming both are in similar timeframes)
        common_start = max(asset_df.index[0], dxy_df.index[0])
        asset_aligned = asset_df[asset_df.index >= common_start]
        dxy_aligned = dxy_df[dxy_df.index >= common_start]

        # Calculate percentage changes
        asset_returns = asset_aligned['Close'].pct_change().dropna()
        dxy_returns = dxy_aligned['Close'].pct_change().dropna()

        # Find common dates
        common_dates = asset_returns.index.intersection(dxy_returns.index)
        asset_common = asset_returns.loc[common_dates]
        dxy_common = dxy_returns.loc[common_dates]

        # Calculate correlation
        correlation = asset_common.corr(dxy_common)

        return {
            'correlation': correlation,
            'sample_size': len(common_dates),
            'asset_returns': asset_common,
            'dxy_returns': dxy_common
        }
    except Exception as e:
        print(f"Error analyzing DXY correlation: {str(e)}")
        return None

def display_signal_panel(signal_data, latest_candle, model_type):
    """Display the trading signal panel"""
    st.subheader(f"🎯 Trading Signal ({model_type})")

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
    elif market_type == "Crypto":
        return fetcher.get_btc_dominance(), fetcher.get_crypto_news()
    else:  # Forex
        return None, fetcher.get_forex_news()

# Initialize variables to avoid undefined errors
df_with_indicators = None
all_signals = {}
all_forecasts = {}

# Fetch and process data first
try:
    with st.spinner("Fetching market data..."):
        df = fetch_market_data(asset, timeframe)

    if df is not None and not df.empty:
        st.success(f"✅ Successfully fetched {len(df)} data points for {asset_display}")
        # Calculate technical indicators
        df_with_indicators = calculate_indicators(df)
    else:
        # Try clearing cache and retrying once
        st.warning("Initial data fetch failed. Retrying...")
        fetch_market_data.clear()  # Clear cache
        df = fetch_market_data(asset, timeframe)  # Retry

        if df is not None and not df.empty:
            st.success(f"✅ Successfully fetched {len(df)} data points for {asset_display} on retry")
            df_with_indicators = calculate_indicators(df)
        else:
            # This block is now handled above with retry logic
            pass
            df_with_indicators = None

        # Generate signals using different models for comparison
        model_types = ["RL", "LSTM", "AutoML", "ML"]
        all_signals = {}
        all_forecasts = {}

        forecast_minutes = forecast_durations[forecast_duration]

        for model_type in model_types:
            try:
                model_inference = ModelInference(model_type=model_type)
                signal_data = model_inference.generate_signal(df_with_indicators, asset)

                # Generate forecast for this model
                forecast_prices = forecast_next_period(df_with_indicators, forecast_minutes, interval_seconds=0.30)
                all_signals[model_type] = signal_data
                all_forecasts[model_type] = forecast_prices

                # Store signal data for chart enhancement
                if signal_data:
                    all_signals[model_type] = signal_data

            except Exception as e:
                print(f"Error with {model_type} model: {str(e)}")
                all_signals[model_type] = None
                all_forecasts[model_type] = None

except Exception as e:
    st.error(f"Error loading data: {str(e)}")

# Main dashboard layout - Adjusted to accommodate live prices at top
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 {asset_display} - {timeframe} Analysis & Forecasting")

    if df_with_indicators is not None and not df_with_indicators.empty:
        # Create the candlestick chart
        fig = create_candlestick_chart(df_with_indicators, asset_display)
        st.plotly_chart(fig, use_container_width=True, key="candlestick_chart")

        # Display current model signal prominently
        current_signal = all_signals.get(st.session_state.model_type)
        if current_signal:
            display_signal_panel(current_signal, df_with_indicators.iloc[-1], st.session_state.model_type)

        # Add current live price comparison for selected asset
        if asset in live_prices:
            current_live_price = live_prices[asset]
            current_historical = df_with_indicators['Close'].iloc[-1]
            price_diff = current_live_price - current_historical
            price_diff_pct = (price_diff / current_historical) * 100 if current_historical != 0 else 0

            st.info(f"📊 **Live Price Update**: {asset_display} is currently ${current_live_price:.4f} "
                   f"({'+' if price_diff >= 0 else ''}{price_diff:.4f} / {price_diff_pct:+.2f}%) "
                   f"from last historical data point")

        # Show multi-model comparison with enhanced precision
        precision_text = "0.30s precision" if forecast_durations[forecast_duration] <= 60 else "1min precision"
        st.header(f"Multi-Model {forecast_duration} Forecast ({precision_text})")

        comparison_chart = create_multi_model_comparison_chart(
            df_with_indicators, asset_display, all_forecasts, forecast_durations[forecast_duration]
        )
        st.plotly_chart(comparison_chart, use_container_width=True, key="multi_model_forecast_chart")

        # Individual model forecast charts with trading levels
        st.subheader("🎯 Individual Model Forecasts with Trading Levels")

        model_colors = {
            "RL": {"bg": "linear-gradient(135deg, #1e3c72, #2a5298)", "text": "🤖 RL Model"},
            "LSTM": {"bg": "linear-gradient(135deg, #667eea, #764ba2)", "text": "🧠 LSTM Model"},
            "AutoML": {"bg": "linear-gradient(135deg, #f093fb, #f5576c)", "text": "⚡ AutoML Model"},
            "ML": {"bg": "linear-gradient(135deg, #4facfe, #00f2fe)", "text": "🎯 ML Model"}
        }

        for model_type in model_types:
            if all_forecasts.get(model_type):
                signal = all_signals.get(model_type)

                # Create a styled container for each model
                st.markdown(f"""
                <div style="background: {model_colors[model_type]['bg']}; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border: 2px solid #00ff88;">
                    <h4 style="color: white; margin: 0;">{model_colors[model_type]['text']} - {forecast_duration} Forecast</h4>
                </div>
                """, unsafe_allow_html=True)

                # Create forecast chart with trading levels
                forecast_chart = create_forecast_chart(
                    df_with_indicators, asset_display, all_forecasts[model_type],
                    forecast_durations[forecast_duration], interval_seconds=0.30,
                    model_type=model_type, signal_data=signal
                )
                st.plotly_chart(forecast_chart, use_container_width=True, key=f"forecast_chart_{model_type}")

                # Display trading levels in a compact format
                if signal:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        signal_type = signal.get('signal', 'NO TRADE')
                        color_class = "signal-buy" if signal_type == 'BUY' else "signal-sell" if signal_type == 'SELL' else "signal-neutral"
                        st.markdown(f'<div class="{color_class}">{signal_type}</div>', unsafe_allow_html=True)

                    with col2:
                        entry = signal.get('entry', df_with_indicators['Close'].iloc[-1])
                        st.metric("Entry", f"{entry:.4f}")

                    with col3:
                        stop_loss = signal.get('stop_loss', 0)
                        if stop_loss > 0:
                            st.metric("Stop Loss", f"{stop_loss:.4f}")
                        else:
                            st.metric("Stop Loss", "N/A")

                    with col4:
                        take_profit = signal.get('take_profit', 0)
                        if take_profit > 0:
                            st.metric("Take Profit", f"{take_profit:.4f}")
                        else:
                            st.metric("Take Profit", "N/A")

                    # Risk/Reward calculation
                    if stop_loss > 0 and take_profit > 0:
                        risk = abs(entry - stop_loss)
                        reward = abs(take_profit - entry)
                        rr_ratio = reward / risk if risk > 0 else 0
                        st.info(f"**Risk:Reward Ratio: 1:{rr_ratio:.2f}** | Risk: ${risk:.4f} | Reward: ${reward:.4f}")

        # Add 1-hour forecast section for longer-term view
        if forecast_durations[forecast_duration] < 60:
            st.subheader("🔮 1-Hour Forecast Overview with Trading Levels")
            try:
                # Generate 1-hour forecasts for all models
                hour_forecasts = {}
                for model_type in model_types:
                    try:
                        hour_forecast = forecast_next_period(df_with_indicators, 60, interval_seconds=60)
                        hour_forecasts[model_type] = hour_forecast
                    except Exception as e:
                        print(f"Error generating 1-hour forecast for {model_type}: {str(e)}")
                        hour_forecasts[model_type] = None

                # Create enhanced 1-hour forecast chart with trading levels
                fig_hour = go.Figure()

                # Historical data (last 2 hours)
                hist_data = df_with_indicators.tail(120) if len(df_with_indicators) >= 120 else df_with_indicators
                fig_hour.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        name='Historical (2h)',
                        line=dict(color='#00ff88', width=3),
                        mode='lines',
                        fill='tozeroy',
                        fillcolor='rgba(0,255,136,0.1)'
                    )
                )

                # Add 1-hour forecasts for each model with enhanced styling
                colors = {'RL': '#ff4444', 'LSTM': '#44ff44', 'AutoML': '#ffaa00', 'ML': '#aa44ff'}
                for model_type, forecasts in hour_forecasts.items():
                    if forecasts:
                        forecast_times = [df_with_indicators.index[-1] + pd.Timedelta(minutes=i+1) for i in range(len(forecasts))]
                        signal = all_signals.get(model_type)

                        # Add forecast line
                        fig_hour.add_trace(
                            go.Scatter(
                                x=forecast_times,
                                y=forecasts,
                                name=f'{model_type} 1h Forecast',
                                mode='lines',
                                line=dict(color=colors.get(model_type, '#888888'), width=3, dash='solid')
                            )
                        )

                        # Add trading levels if signal exists
                        if signal:
                            entry_price = signal.get('entry', df_with_indicators['Close'].iloc[-1])
                            stop_loss = signal.get('stop_loss', 0)
                            take_profit = signal.get('take_profit', 0)

                            # Entry level (horizontal line across entire chart)
                            fig_hour.add_hline(y=entry_price, line_dash="solid", line_color=colors.get(model_type, '#888888'),
                                             line_width=2, opacity=0.7)

                            # Stop Loss level
                            if stop_loss > 0:
                                fig_hour.add_hline(y=stop_loss, line_dash="dash", line_color="#ff4444",
                                                 line_width=2, opacity=0.7)

                            # Take Profit level
                            if take_profit > 0:
                                fig_hour.add_hline(y=take_profit, line_dash="dot", line_color="#44ff44",
                                                 line_width=2, opacity=0.7)

                # Add current price marker
                current_price = df_with_indicators['Close'].iloc[-1]
                fig_hour.add_hline(y=current_price, line_dash="solid", line_color="#00ff88", line_width=4,
                                 annotation_text=f"Current: ${current_price:.4f}", annotation_position="top right")

                fig_hour.update_layout(
                    title={
                        'text': f'1-Hour Forecast Overview - {asset_display} (with Trading Levels)',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=18, color='#00ff88', weight='bold')
                    },
                    height=400,
                    showlegend=True,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0.95)',
                    paper_bgcolor='rgba(0,0,0,0.98)',
                    font=dict(color='white', size=12),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        linecolor='rgba(255,255,255,0.3)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        linecolor='rgba(255,255,255,0.3)'
                    )
                )

                st.plotly_chart(fig_hour, use_container_width=True, key="hour_forecast_chart")

                # Add summary of 1-hour trading levels
                st.subheader("📊 1-Hour Trading Levels Summary")
                summary_data = []
                for model_type in model_types:
                    signal = all_signals.get(model_type)
                    forecast = hour_forecasts.get(model_type)
                    if signal and forecast:
                        entry = signal.get('entry', df_with_indicators['Close'].iloc[-1])
                        stop_loss = signal.get('stop_loss', 0)
                        take_profit = signal.get('take_profit', 0)
                        forecast_end = forecast[-1]

                        risk = abs(entry - stop_loss) if stop_loss > 0 else 0
                        reward = abs(take_profit - entry) if take_profit > 0 else 0
                        rr_ratio = reward / risk if risk > 0 else 0

                        summary_data.append({
                            'Model': model_type,
                            'Signal': signal.get('signal', 'N/A'),
                            'Entry': f"${entry:.4f}",
                            'Stop Loss': f"${stop_loss:.4f}" if stop_loss > 0 else "N/A",
                            'Take Profit': f"${take_profit:.4f}" if take_profit > 0 else "N/A",
                            '1H Forecast': f"${forecast_end:.4f}",
                            'Risk:Reward': f"1:{rr_ratio:.2f}" if rr_ratio > 0 else "N/A"
                        })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"Unable to generate 1-hour forecast: {str(e)}")

        # Display forecast info for all models
        st.subheader("Model Performance Metrics")

        # Create metrics table
        metrics_data = []
        for model_type in model_types:
            signal = all_signals.get(model_type)
            forecast = all_forecasts.get(model_type)

            if signal and forecast:
                # Get accuracy info for current model
                if model_type == st.session_state.model_type:
                    accuracy_info = model_inference.get_forecast_accuracy(asset)
                else:
                    # Create temporary inference object for other models
                    temp_inference = ModelInference(model_type=model_type)
                    accuracy_info = temp_inference.get_forecast_accuracy(asset)

                metrics_data.append({
                    'Model': model_type,
                    'Signal': signal['signal'],
                    'Confidence': f"{signal['confidence']:.1f}%",
                    'Current Price': f"{df_with_indicators['Close'].iloc[-1]:.4f}",
                    'Forecast Price': f"{forecast[-1]:.4f}",
                    'Accuracy': f"{accuracy_info['accuracy_score']:.2f}",
                    'Penalty': f"{accuracy_info['penalty']:.1f}%"
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

        # Live market comparison section
        st.subheader("Live Market Comparison")

        # Get current live price for comparison
        try:
            live_fetcher = DataFetcher()
            live_data = live_fetcher.get_ohlcv_data(asset, timeframe, periods=1)  # Get latest data

            if live_data is not None and not live_data.empty:
                live_price = live_data['Close'].iloc[-1]
                current_forecast = all_forecasts.get(st.session_state.model_type, [])

                if current_forecast:
                    forecasted_price = current_forecast[-1]
                    price_diff = forecasted_price - live_price
                    price_diff_pct = (price_diff / live_price) * 100

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Live Market Price", f"{live_price:.4f}")

                    with col2:
                        st.metric(f"{st.session_state.model_type} Forecast", f"{forecasted_price:.4f}")

                    with col3:
                        delta_color = "normal" if abs(price_diff_pct) < 2 else ("inverse" if price_diff_pct > 0 else "normal")
                        st.metric("Forecast vs Live", f"{price_diff_pct:+.2f}%",
                                delta=f"{price_diff:+.4f}", delta_color=delta_color)

                    # Update forecast accuracy tracking
                    forecast_timestamp = datetime.now() - timedelta(minutes=forecast_minutes)
                    model_inference.update_forecast_accuracy(asset, forecasted_price, live_price, forecast_timestamp)

                    # Show accuracy trend
                    accuracy_info = model_inference.get_forecast_accuracy(asset)
                    st.info(f"Model Accuracy: {accuracy_info['accuracy_score']:.2f} | "
                           f"Penalty Applied: {accuracy_info['penalty']:.1f}% | "
                           f"Forecasts Tracked: {accuracy_info['forecast_count']}")

        except Exception as e:
            st.warning(f"Unable to fetch live market data for comparison: {str(e)}")

    else:
        st.error(f"❌ Unable to fetch market data for {asset_display} ({asset}). Please check the symbol or try again later.")
        st.info("💡 **Troubleshooting tips:**")
        st.info("• Check if the market is open (weekdays, business hours)")
        st.info("• Try selecting a different asset or timeframe")
        st.info("• Refresh the page to retry data fetching")

with col2:
    st.subheader("📈 Fundamentals")

    try:
        fund_data1, fund_data2 = get_fundamentals_data(market_type)

        if market_type == "Gold":
            # Display DXY chart
            st.write("**💵 Dollar Index (DXY)**")
            if fund_data1 is not None and hasattr(fund_data1, 'empty') and not fund_data1.empty:
                dxy_fig = create_line_chart(fund_data1, "DXY", "Dollar Index")
                st.plotly_chart(dxy_fig, use_container_width=True, key="dxy_line_chart")

                # DXY impact analysis
                if df_with_indicators is not None:
                    dxy_correlation = analyze_dxy_correlation(df_with_indicators, fund_data1)
                    if dxy_correlation:
                        corr_value = dxy_correlation.get('correlation', 0)
                        impact_strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
                        corr_direction = "Negative" if corr_value < 0 else "Positive"
                        st.info(f"DXY Correlation: {corr_value:.2f} ({impact_strength} {corr_direction})")


            else:
                st.info("DXY data unavailable")

            # Display forex news
            st.write("**📰 Economic Events**")
            if fund_data2:
                for i, news in enumerate(fund_data2[:5]):
                    st.write(f"• {news}")
            else:
                st.info("No recent economic events")

        elif market_type == "Crypto":
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

        else:  # Forex
            # Display forex news
            st.write("**📰 Forex News & Events**")
            if fund_data2:
                for i, news in enumerate(fund_data2[:5]):
                    st.write(f"• {news}")
            else:
                st.info("No recent forex news")

    except Exception as e:
        st.error(f"Error loading fundamentals: {str(e)}")

# Technical indicators panel (only show if we have data)
if df_with_indicators is not None and not df_with_indicators.empty:
    try:
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
            trend = "Bullish" if ema20 > ema50 else "Bearish"
            st.metric("EMA Trend", trend)

        with col3:
            macd = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            st.metric("MACD", macd_trend)

        with col4:
            atr = latest.get('ATR_14', 0)
            st.metric("ATR(14)", f"{atr:.4f}")
    except Exception as e:
        st.error(f"Error displaying technical indicators: {str(e)}")

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

# Health check endpoint for monitoring
def health_check():
    """Simple health check function"""
    try:
        # Test if data fetching works
        fetcher = DataFetcher()
        test_data = fetcher.get_ohlcv_data('GC=F', '5m', periods=5)
        data_status = "data_fetch_ok" if test_data is not None and not test_data.empty else "data_fetch_failed"

        # Test model inference
        model_status = "model_ok"
        try:
            model_inference = ModelInference(model_type="ML")
            test_signal = model_inference.generate_signal(test_data, 'GC=F') if test_data is not None else None
            if test_signal is None:
                model_status = "model_failed"
        except Exception as model_e:
            model_status = f"model_error: {str(model_e)}"

        return {
            "status": "healthy" if data_status == "data_fetch_ok" and model_status == "model_ok" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "data_fetch": data_status,
            "model_inference": model_status,
            "model_types": ["RL", "LSTM", "AutoML", "ML"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Create a simple health check page
def show_health_page():
    """Display health check information"""
    st.title("🔍 Health Check Dashboard")

    health_info = health_check()

    # Status indicator
    if health_info["status"] == "healthy":
        st.success("✅ System Status: HEALTHY")
    elif health_info["status"] == "degraded":
        st.warning("⚠️ System Status: DEGRADED")
    else:
        st.error("❌ System Status: UNHEALTHY")

    # Health metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Fetch", health_info.get("data_fetch", "unknown"))

    with col2:
        st.metric("Model Inference", health_info.get("model_inference", "unknown"))

    with col3:
        st.metric("Last Check", health_info.get("timestamp", "unknown")[:19])

    # Model types
    st.subheader("🤖 Available Models")
    models = health_info.get("model_types", [])
    for model in models:
        st.write(f"• {model}")

    # Error details if any
    if "error" in health_info:
        st.error(f"Error: {health_info['error']}")

# Check if health page is requested
if st.query_params.get("page") == "health":
    show_health_page()
    st.stop()

# Add health check to Streamlit if needed
if __name__ == "__main__":
    # This ensures the app runs properly when executed directly
    pass
