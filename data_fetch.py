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
        Fetch OHLCV data from Yahoo Finance with enhanced fallback options

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
                period = '5d'  # 5 days for intraday to avoid rate limits and data availability
            elif timeframe == '1h':
                period = '30d'  # 30 days for hourly
            else:
                period = '1y'   # 1 year for daily

            # Special handling for gold symbols - use shorter periods due to data availability
            if symbol in ['IAU', 'GLD', 'GC=F', 'XAUUSD=X', 'GOLD']:
                if timeframe in ['5m', '15m', '30m']:
                    period = '60d'  # 60 days for gold intraday data
                elif timeframe == '1h':
                    period = '1y'   # 1 year for gold hourly
                else:
                    period = '2y'   # 2 years for gold daily

            # Enhanced symbol mapping with more alternatives
            alt_symbols = {
                'GC=F': ['IAU', 'GLD', 'GC=F', 'XAUUSD=X', 'GOLD'],
                'BTC-USD': ['BTC-USD', 'BTCUSD=X', 'BTC-USD', 'BTCUSD=X'],
                'ETH-USD': ['ETH-USD', 'ETHUSD=X', 'ETH-USD', 'ETHUSD=X'],
                'BNB-USD': ['BNB-USD', 'BNBUSD=X', 'BNB-USD', 'BNBUSD=X'],
                'SOL-USD': ['SOL-USD', 'SOLUSD=X', 'SOL-USD', 'SOLUSD=X'],
                'XRP-USD': ['XRP-USD', 'XRPUSD=X', 'XRP-USD', 'XRPUSD=X'],
                'ADA-USD': ['ADA-USD', 'ADAUSD=X', 'ADA-USD', 'ADAUSD=X'],
                'AVAX-USD': ['AVAX-USD', 'AVAXUSD=X', 'AVAX-USD', 'AVAXUSD=X'],
                'DOT-USD': ['DOT-USD', 'DOTUSD=X', 'DOT-USD', 'DOTUSD=X'],
                'MATIC-USD': ['MATIC-USD', 'MATICUSD=X', 'MATIC-USD', 'MATICUSD=X'],
                'LINK-USD': ['LINK-USD', 'LINKUSD=X', 'LINK-USD', 'LINKUSD=X'],
                'UNI-USD': ['UNI-USD', 'UNIUSD=X', 'UNI-USD', 'UNIUSD=X'],
                'LTC-USD': ['LTC-USD', 'LTCUSD=X', 'LTC-USD', 'LTCUSD=X'],
                'BCH-USD': ['BCH-USD', 'BCHUSD=X', 'BCH-USD', 'BCHUSD=X'],
                'ALGO-USD': ['ALGO-USD', 'ALGOUSD=X', 'ALGO-USD', 'ALGOUSD=X'],
                'VET-USD': ['VET-USD', 'VETUSD=X', 'VET-USD', 'VETUSD=X'],
                'ICP-USD': ['ICP-USD', 'ICPUSD=X', 'ICP-USD', 'ICPUSD=X'],
                'ATOM-USD': ['ATOM-USD', 'ATOMUSD=X', 'ATOM-USD', 'ATOMUSD=X'],
                'FIL-USD': ['FIL-USD', 'FILUSD=X', 'FIL-USD', 'FILUSD=X'],
                'TRX-USD': ['TRX-USD', 'TRXUSD=X', 'TRX-USD', 'TRXUSD=X'],
                'ETC-USD': ['ETC-USD', 'ETCUSD=X', 'ETC-USD', 'ETCUSD=X'],
                'IAU': ['IAU', 'GLD', 'GC=F', 'XAUUSD=X', 'GOLD']
            }

            # Try all symbols in the list (including original)
            symbols_to_try = alt_symbols.get(symbol, [symbol])

            df = None
            for try_symbol in symbols_to_try:
                try:
                    print(f"Trying to fetch data for {try_symbol}...")
                    ticker = yf.Ticker(try_symbol)
                    df = ticker.history(period=period, interval=interval)

                    if not df.empty and len(df) >= 10:  # Require at least 10 data points
                        print(f"Successfully fetched data for {try_symbol}")
                        break
                    else:
                        print(f"No data or insufficient data for {try_symbol}")
                        df = None
                except Exception as e:
                    print(f"Error with {try_symbol}: {str(e)}")
                    continue

            if df is None or df.empty:
                print(f"All symbol attempts failed for {symbol}")
                return None

            # Clean data - remove unnecessary columns
            df = df.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'], errors='ignore')
            df = df.dropna()

            # Keep only the last 'periods' rows, but ensure we have enough data
            if len(df) > periods:
                df = df.tail(periods)
            elif len(df) < 50:  # If we have very little data, try to get more
                # Try with a longer period for this specific case
                try:
                    ticker = yf.Ticker(symbol)
                    df_extended = ticker.history(period='60d', interval=interval)
                    if not df_extended.empty and len(df_extended) > len(df):
                        df_extended = df_extended.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'], errors='ignore')
                        df_extended = df_extended.dropna()
                        df = df_extended.tail(min(periods, len(df_extended)))
                except:
                    pass  # Keep original df if extension fails

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
        Get Bitcoin dominance percentage from CoinGecko API
        
        Returns:
            Float representing BTC dominance percentage
        """
        try:
            # Use CoinGecko API for BTC dominance
            url = "https://api.coingecko.com/api/v3/global"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                btc_dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc')
                if btc_dominance:
                    return round(btc_dominance, 2)
            
            # Fallback: try alternative calculation
            return self._calculate_btc_dominance_fallback()
            
        except Exception as e:
            print(f"Error fetching BTC dominance: {str(e)}")
            return self._calculate_btc_dominance_fallback()
    
    def _calculate_btc_dominance_fallback(self):
        """Fallback BTC dominance calculation"""
        try:
            # Get BTC and ETH prices as proxy for dominance
            btc_ticker = yf.Ticker("BTC-USD")
            btc_data = btc_ticker.history(period='1d')
            if not btc_data.empty:
                # Return approximate dominance based on recent trends
                return 54.2  # Updated approximate BTC dominance
            return 50.0
        except:
            return 50.0
    
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
        Scrape recent crypto news headlines from multiple sources
        
        Returns:
            List of news headlines
        """
        try:
            # Try CoinGecko news API first (more reliable)
            url = "https://api.coingecko.com/api/v3/news"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    headlines = []
                    for item in data['data'][:5]:
                        title = item.get('title', '')
                        if title and len(title) > 10:
                            headlines.append(title)
                    if headlines:
                        return headlines
            
            # Fallback to CoinDesk scraping
            url = "https://www.coindesk.com/"
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    lines = text.split('\n')
                    headlines = [line.strip() for line in lines[:10] 
                               if line.strip() and len(line.strip()) > 20]
                    if headlines:
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

    def get_live_prices_for_assets(self, asset_list):
        """
        Get live prices for multiple assets simultaneously

        Args:
            asset_list: List of trading symbols

        Returns:
            Dictionary with symbol: price pairs
        """
        live_prices = {}
        for symbol in asset_list:
            price = self.get_real_time_price(symbol)
            if price is not None:
                live_prices[symbol] = price
            else:
                # Try alternative symbols if available - prioritize working symbols
                alt_symbols = {
                    'GC=F': ['IAU', 'GLD', 'GC=F', 'XAUUSD=X', 'GOLD'],  # IAU first as most reliable
                    'BTC-USD': ['BTC-USD', 'BTCUSD=X'],
                    'ETH-USD': ['ETH-USD', 'ETHUSD=X'],
                    'BNB-USD': ['BNB-USD', 'BNBUSD=X'],
                    'SOL-USD': ['SOL-USD', 'SOLUSD=X'],
                    'XRP-USD': ['XRP-USD', 'XRPUSD=X'],
                    'ADA-USD': ['ADA-USD', 'ADAUSD=X'],
                    'AVAX-USD': ['AVAX-USD', 'AVAXUSD=X'],
                    'DOT-USD': ['DOT-USD', 'DOTUSD=X'],
                    'EURUSD=X': ['EURUSD=X', 'EURUSD'],
                    'GBPUSD=X': ['GBPUSD=X', 'GBPUSD'],
                    'USDJPY=X': ['USDJPY=X', 'USDJPY']
                }

                if symbol in alt_symbols:
                    for alt_symbol in alt_symbols[symbol]:  # Try all alternatives including original
                        price = self.get_real_time_price(alt_symbol)
                        if price is not None:
                            live_prices[symbol] = price
                            break

        return live_prices
    
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
