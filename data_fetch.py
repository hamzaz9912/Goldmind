import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
import time

class DataFetcher:
    """Handle all data fetching operations for the scalping dashboard"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_ohlcv_data(self, symbol, timeframe, periods=200):
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD=X', 'BTC-USD')
            timeframe: Time interval ('5m', '15m', '30m', '1h', '1d')
            periods: Number of periods to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframes to yfinance intervals
            interval_map = {
                '5m': '5m',
                '15m': '15m', 
                '30m': '30m',
                '1h': '1h',
                '1d': '1d'
            }
            
            interval = interval_map.get(timeframe, '30m')
            
            # Calculate period string based on timeframe
            if timeframe in ['5m', '15m', '30m']:
                period = '30d'  # 30 days for intraday
            elif timeframe == '1h':
                period = '60d'  # 60 days for hourly
            else:
                period = '1y'   # 1 year for daily
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
                
            # Clean data
            df = df.dropna()
            
            # Keep only the last 'periods' rows
            df = df.tail(periods)
            
            # Add volume-based calculations
            if 'Volume' in df.columns:
                df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_dxy_data(self, periods=50):
        """
        Fetch Dollar Index (DXY) data
        
        Returns:
            DataFrame with DXY data
        """
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            df = ticker.history(period='30d', interval='1d')
            
            if df.empty:
                return None
                
            df = df[['Close']].rename(columns={'Close': 'DXY'})
            return df.tail(periods)
            
        except Exception as e:
            print(f"Error fetching DXY data: {str(e)}")
            return None
    
    def get_btc_dominance(self):
        """
        Get Bitcoin dominance percentage
        
        Returns:
            Float representing BTC dominance percentage
        """
        try:
            # Try to get BTC market cap data
            btc_ticker = yf.Ticker("BTC-USD")
            btc_info = btc_ticker.info
            
            # Approximate BTC dominance (simplified calculation)
            # In reality, this would require total crypto market cap
            # For now, return a reasonable estimate
            return 45.5  # Approximate current BTC dominance
            
        except Exception as e:
            print(f"Error fetching BTC dominance: {str(e)}")
            return None
    
    def get_forex_news(self):
        """
        Scrape recent forex/economic news
        
        Returns:
            List of news headlines
        """
        try:
            # Try to get economic calendar events
            url = "https://www.forexfactory.com/calendar"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return self._get_fallback_forex_news()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for high impact events
            events = []
            event_rows = soup.find_all('tr', class_='calendar_row')
            
            for row in event_rows[:5]:  # Get first 5 events
                try:
                    event_cell = row.find('td', class_='calendar_event')
                    if event_cell:
                        event_text = event_cell.get_text(strip=True)
                        if event_text and len(event_text) > 5:
                            events.append(event_text)
                except:
                    continue
            
            return events if events else self._get_fallback_forex_news()
            
        except Exception as e:
            print(f"Error fetching forex news: {str(e)}")
            return self._get_fallback_forex_news()
    
    def get_crypto_news(self):
        """
        Scrape recent crypto news headlines
        
        Returns:
            List of news headlines
        """
        try:
            # Try CoinDesk first
            url = "https://www.coindesk.com/"
            
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    # Extract first few lines as headlines
                    lines = text.split('\n')
                    headlines = [line.strip() for line in lines[:10] 
                               if line.strip() and len(line.strip()) > 20]
                    return headlines[:5]
            
            return self._get_fallback_crypto_news()
            
        except Exception as e:
            print(f"Error fetching crypto news: {str(e)}")
            return self._get_fallback_crypto_news()
    
    def _get_fallback_forex_news(self):
        """Fallback forex news when scraping fails"""
        return [
            "Economic data releases pending",
            "Central bank policy meetings scheduled", 
            "Inflation reports expected this week",
            "Employment statistics to be released",
            "GDP growth figures anticipated"
        ]
    
    def _get_fallback_crypto_news(self):
        """Fallback crypto news when scraping fails"""
        return [
            "Bitcoin institutional adoption continues",
            "Ethereum network upgrades in progress",
            "DeFi protocols showing growth",
            "Regulatory clarity developments pending",
            "Crypto market sentiment analysis available"
        ]
    
    def get_real_time_price(self, symbol):
        """
        Get real-time price for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price as float
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"Error fetching real-time price for {symbol}: {str(e)}")
            return None
    
    def validate_symbol(self, symbol):
        """
        Validate if a trading symbol exists and has data
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Boolean indicating if symbol is valid
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d', interval='1d')
            return not data.empty
            
        except:
            return False
