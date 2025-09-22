import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def format_number(number, decimal_places=4):
    """
    Format number for display
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    try:
        if pd.isna(number) or number == 0:
            return "0.0000"
        return f"{float(number):.{decimal_places}f}"
    except:
        return "N/A"

def format_percentage(number):
    """
    Format percentage for display
    
    Args:
        number: Number to format as percentage
        
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(number):
            return "N/A"
        return f"{float(number):.2f}%"
    except:
        return "N/A"

def calculate_risk_reward_ratio(entry, stop_loss, take_profit):
    """
    Calculate risk-reward ratio
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        Risk-reward ratio as string
    """
    try:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return "N/A"
        
        ratio = reward / risk
        return f"1:{ratio:.2f}"
    except:
        return "N/A"

def get_signal_color(signal):
    """
    Get color for signal display
    
    Args:
        signal: Signal string ('BUY', 'SELL', 'NO TRADE')
        
    Returns:
        Color string
    """
    color_map = {
        'BUY': '#00ff88',
        'SELL': '#ff4444', 
        'NO TRADE': '#888888'
    }
    return color_map.get(signal, '#888888')

def get_signal_emoji(signal):
    """
    Get emoji for signal display
    
    Args:
        signal: Signal string
        
    Returns:
        Emoji string
    """
    emoji_map = {
        'BUY': '🟢',
        'SELL': '🔴',
        'NO TRADE': '⚪'
    }
    return emoji_map.get(signal, '⚪')

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame for required columns and data
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
    
    return True

def calculate_percentage_change(current, previous):
    """
    Calculate percentage change between two values
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Percentage change
    """
    try:
        if previous == 0:
            return 0
        return ((current - previous) / previous) * 100
    except:
        return 0

def get_timeframe_minutes(timeframe):
    """
    Convert timeframe string to minutes
    
    Args:
        timeframe: Timeframe string ('5m', '15m', etc.)
        
    Returns:
        Number of minutes
    """
    timeframe_map = {
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '1d': 1440
    }
    return timeframe_map.get(timeframe, 30)

def format_timestamp(timestamp):
    """
    Format timestamp for display
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "N/A"

def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss):
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Account balance
        risk_percentage: Risk percentage (e.g., 2 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        
    Returns:
        Position size
    """
    try:
        risk_amount = account_balance * (risk_percentage / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    except:
        return 0

def get_market_session():
    """
    Determine current market session based on UTC time
    
    Returns:
        Market session string
    """
    try:
        utc_hour = datetime.utcnow().hour
        
        # Market sessions (UTC)
        if 22 <= utc_hour or utc_hour < 7:
            return "Asian Session"
        elif 7 <= utc_hour < 15:
            return "London Session" 
        elif 15 <= utc_hour < 22:
            return "New York Session"
        else:
            return "Market Closed"
    except:
        return "Unknown"

def is_market_open(symbol):
    """
    Check if market is open for a given symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Boolean indicating if market is open
    """
    try:
        current_time = datetime.utcnow()
        weekday = current_time.weekday()
        
        # Crypto markets are always open
        if '-USD' in symbol:
            return True
        
        # Forex/Gold markets (Monday-Friday)
        if weekday < 5:  # Monday = 0, Friday = 4
            return True
        elif weekday == 6:  # Sunday
            # Markets open Sunday 22:00 UTC
            return current_time.hour >= 22
        else:  # Saturday
            return False
    except:
        return True  # Default to open

def clean_symbol_for_filename(symbol):
    """
    Clean symbol string for use in filenames
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Cleaned symbol string
    """
    return symbol.replace('=', '_').replace('-', '_').replace('/', '_')

def get_asset_display_name(symbol):
    """
    Get human-readable display name for asset
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Display name string
    """
    display_names = {
        'XAUUSD=X': 'Gold (XAUUSD)',
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'BNB-USD': 'Binance Coin',
        'SOL-USD': 'Solana',
        'XRP-USD': 'Ripple',
        'ADA-USD': 'Cardano',
        'AVAX-USD': 'Avalanche',
        'DOT-USD': 'Polkadot',
        'MATIC-USD': 'Polygon',
        'LINK-USD': 'Chainlink',
        'UNI-USD': 'Uniswap',
        'LTC-USD': 'Litecoin',
        'BCH-USD': 'Bitcoin Cash',
        'ALGO-USD': 'Algorand',
        'VET-USD': 'VeChain',
        'ICP-USD': 'Internet Computer',
        'ATOM-USD': 'Cosmos',
        'FIL-USD': 'Filecoin',
        'TRX-USD': 'TRON',
        'ETC-USD': 'Ethereum Classic'
    }
    
    return display_names.get(symbol, symbol)

def create_signal_summary(signals):
    """
    Create summary of multiple signals
    
    Args:
        signals: Dict of signals
        
    Returns:
        Summary statistics
    """
    try:
        if not signals:
            return {}
        
        buy_count = sum(1 for s in signals.values() if s.get('signal') == 'BUY')
        sell_count = sum(1 for s in signals.values() if s.get('signal') == 'SELL')
        no_trade_count = sum(1 for s in signals.values() if s.get('signal') == 'NO TRADE')
        
        avg_confidence = np.mean([s.get('confidence', 0) for s in signals.values()])
        
        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'no_trade_signals': no_trade_count,
            'average_confidence': avg_confidence,
            'buy_percentage': (buy_count / len(signals)) * 100,
            'sell_percentage': (sell_count / len(signals)) * 100
        }
    except:
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_data(func, *args, **kwargs):
    """
    Generic caching wrapper for data fetching functions
    
    Args:
        func: Function to cache
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    return func(*args, **kwargs)

def log_error(error_message, context=""):
    """
    Log error message with timestamp
    
    Args:
        error_message: Error message to log
        context: Additional context information
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR: {error_message}"
    if context:
        log_entry += f" | Context: {context}"
    
    print(log_entry)
    
    # Could extend to write to file or external logging service
    # with open('app.log', 'a') as f:
    #     f.write(log_entry + '\n')

def check_data_freshness(timestamp, max_age_minutes=30):
    """
    Check if data is fresh enough
    
    Args:
        timestamp: Data timestamp
        max_age_minutes: Maximum age in minutes
        
    Returns:
        Boolean indicating if data is fresh
    """
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        age = datetime.now() - timestamp
        return age.total_seconds() < (max_age_minutes * 60)
    except:
        return False
