import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""

    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df):
        """
        Calculate all technical indicators needed for the system
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if df is None or df.empty:
            return df
            
        df_copy = df.copy()
        
        try:
            # Trend Indicators
            df_copy = self.add_ema_indicators(df_copy)
            
            # Momentum Indicators  
            df_copy = self.add_rsi(df_copy)
            df_copy = self.add_macd(df_copy)
            
            # Volatility Indicators
            df_copy = self.add_atr(df_copy)
            df_copy = self.add_bollinger_bands(df_copy)
            
            # Volume Indicators (for crypto)
            if 'Volume' in df_copy.columns:
                df_copy = self.add_vwap(df_copy)
                df_copy = self.add_volume_sma(df_copy)
            
            # Support/Resistance
            df_copy = self.add_pivot_points(df_copy)
            
            # Custom composite indicators
            df_copy = self.add_trend_strength(df_copy)
            df_copy = self.add_momentum_score(df_copy)
            
            return df_copy
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df
    
    def add_ema_indicators(self, df):
        """Add Exponential Moving Averages"""
        try:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

            # EMA crossover signals
            df['EMA_Signal'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)

            return df
        except Exception as e:
            print(f"Error calculating EMA: {str(e)}")
            return df
    
    def add_rsi(self, df, period=14):
        """Add Relative Strength Index"""
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

            # RSI signals
            df['RSI_Oversold'] = np.where(df[f'RSI_{period}'] < 30, 1, 0)
            df['RSI_Overbought'] = np.where(df[f'RSI_{period}'] > 70, 1, 0)

            return df
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return df
    
    def add_macd(self, df, fast=12, slow=26, signal=9):
        """Add MACD indicator"""
        try:
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            df[f'MACD_{fast}_{slow}_{signal}'] = macd_line
            df[f'MACDs_{fast}_{slow}_{signal}'] = signal_line
            df[f'MACDh_{fast}_{slow}_{signal}'] = histogram

            # MACD signals
            df['MACD_Signal'] = np.where(macd_line > signal_line, 1, -1)

            return df
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return df
    
    def add_atr(self, df, period=14):
        """Add Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift(1))
            low_close = np.abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'ATR_{period}'] = tr.rolling(window=period).mean()

            # ATR-based support/resistance levels
            df['ATR_Upper'] = df['Close'] + (2 * df[f'ATR_{period}'])
            df['ATR_Lower'] = df['Close'] - (2 * df[f'ATR_{period}'])

            return df
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            return df
    
    def add_bollinger_bands(self, df, period=20, std_dev=2):
        """Add Bollinger Bands"""
        try:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)

            df[f'BB_Upper_{period}_{std_dev}'] = upper
            df[f'BB_Middle_{period}_{std_dev}'] = sma
            df[f'BB_Lower_{period}_{std_dev}'] = lower
            df[f'BB_Width_{period}_{std_dev}'] = (upper - lower) / sma
            df[f'BB_Percent_{period}_{std_dev}'] = (df['Close'] - lower) / (upper - lower)

            return df
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            return df
    
    def add_vwap(self, df):
        """Add Volume Weighted Average Price"""
        try:
            if 'Volume' in df.columns:
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                cumulative_volume = df['Volume'].cumsum()
                cumulative_vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume
                df['VWAP'] = cumulative_vwap

                # VWAP signals
                df['VWAP_Signal'] = np.where(df['Close'] > df['VWAP'], 1, -1)

            return df
        except Exception as e:
            print(f"Error calculating VWAP: {str(e)}")
            return df
    
    def add_volume_sma(self, df, period=20):
        """Add Volume Simple Moving Average"""
        try:
            if 'Volume' in df.columns:
                df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()

                # Volume surge detection
                df['Volume_Surge'] = np.where(
                    df['Volume'] > (1.5 * df[f'Volume_SMA_{period}']), 1, 0
                )

            return df
        except Exception as e:
            print(f"Error calculating Volume SMA: {str(e)}")
            return df
    
    def add_pivot_points(self, df):
        """Add Pivot Points for Support/Resistance"""
        try:
            # Calculate pivot points based on previous day's data
            df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
            df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
            df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
            df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
            
            return df
        except Exception as e:
            print(f"Error calculating Pivot Points: {str(e)}")
            return df
    
    def add_trend_strength(self, df):
        """Add custom trend strength indicator"""
        try:
            # Combine multiple trend indicators
            conditions = []
            
            if 'EMA_Signal' in df.columns:
                conditions.append(df['EMA_Signal'])
            
            if 'MACD_Signal' in df.columns:
                conditions.append(df['MACD_Signal'])
                
            if 'VWAP_Signal' in df.columns:
                conditions.append(df['VWAP_Signal'])
            
            if conditions:
                df['Trend_Strength'] = np.mean(conditions, axis=0)
            else:
                df['Trend_Strength'] = 0
            
            return df
        except Exception as e:
            print(f"Error calculating Trend Strength: {str(e)}")
            return df
    
    def add_momentum_score(self, df):
        """Add custom momentum score"""
        try:
            score = 0
            count = 0
            
            # RSI contribution
            if 'RSI_14' in df.columns:
                rsi_score = np.where(df['RSI_14'] > 50, 1, -1)
                score += rsi_score
                count += 1
            
            # MACD contribution
            if 'MACD_Signal' in df.columns:
                score += df['MACD_Signal']
                count += 1
            
            # Price vs EMA contribution
            if 'EMA_20' in df.columns:
                price_score = np.where(df['Close'] > df['EMA_20'], 1, -1)
                score += price_score
                count += 1
            
            # Normalize score
            if count > 0:
                df['Momentum_Score'] = score / count
            else:
                df['Momentum_Score'] = 0
            
            return df
        except Exception as e:
            print(f"Error calculating Momentum Score: {str(e)}")
            return df
    
    def get_support_resistance_levels(self, df, lookback=20):
        """
        Calculate dynamic support and resistance levels
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of periods to look back
            
        Returns:
            Dict with support and resistance levels
        """
        try:
            if len(df) < lookback:
                return {'support': None, 'resistance': None}
            
            recent_data = df.tail(lookback)
            
            # Simple support/resistance based on recent highs/lows
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            return {
                'support': support,
                'resistance': resistance,
                'pivot': (resistance + support) / 2
            }
            
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {'support': None, 'resistance': None}
    
    def calculate_signal_strength(self, df):
        """
        Calculate overall signal strength based on multiple indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Signal strength score (-1 to 1)
        """
        try:
            if df.empty:
                return 0
                
            latest = df.iloc[-1]
            signals = []
            
            # EMA signal
            if 'EMA_Signal' in latest:
                signals.append(latest['EMA_Signal'])
            
            # MACD signal
            if 'MACD_Signal' in latest:
                signals.append(latest['MACD_Signal'])
            
            # RSI signal
            if 'RSI_14' in latest:
                if latest['RSI_14'] > 70:
                    signals.append(-1)  # Overbought
                elif latest['RSI_14'] < 30:
                    signals.append(1)   # Oversold
                else:
                    signals.append(0)   # Neutral
            
            # Volume signal (if available)
            if 'Volume_Surge' in latest:
                signals.append(latest['Volume_Surge'])
            
            # Calculate average signal
            if signals:
                return np.mean(signals)
            else:
                return 0
                
        except Exception as e:
            print(f"Error calculating signal strength: {str(e)}")
            return 0
