import pandas as pd
import numpy as np
import pandas_ta as ta

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
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['EMA_200'] = ta.ema(df['Close'], length=200)
            
            # EMA crossover signals
            df['EMA_Signal'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
            
            return df
        except Exception as e:
            print(f"Error calculating EMA: {str(e)}")
            return df
    
    def add_rsi(self, df, period=14):
        """Add Relative Strength Index"""
        try:
            df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period)
            
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
            macd_data = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
            
            if macd_data is not None:
                df[f'MACD_{fast}_{slow}_{signal}'] = macd_data[f'MACD_{fast}_{slow}_{signal}']
                df[f'MACDs_{fast}_{slow}_{signal}'] = macd_data[f'MACDs_{fast}_{slow}_{signal}']
                df[f'MACDh_{fast}_{slow}_{signal}'] = macd_data[f'MACDh_{fast}_{slow}_{signal}']
                
                # MACD signals
                df['MACD_Signal'] = np.where(
                    df[f'MACD_{fast}_{slow}_{signal}'] > df[f'MACDs_{fast}_{slow}_{signal}'], 
                    1, -1
                )
            
            return df
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return df
    
    def add_atr(self, df, period=14):
        """Add Average True Range"""
        try:
            df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
            
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
            bb_data = ta.bbands(df['Close'], length=period, std=std_dev)
            
            if bb_data is not None:
                df[f'BB_Upper_{period}_{std_dev}'] = bb_data[f'BBU_{period}_{std_dev}']
                df[f'BB_Middle_{period}_{std_dev}'] = bb_data[f'BBM_{period}_{std_dev}']
                df[f'BB_Lower_{period}_{std_dev}'] = bb_data[f'BBL_{period}_{std_dev}']
                df[f'BB_Width_{period}_{std_dev}'] = bb_data[f'BBB_{period}_{std_dev}']
                df[f'BB_Percent_{period}_{std_dev}'] = bb_data[f'BBP_{period}_{std_dev}']
            
            return df
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            return df
    
    def add_vwap(self, df):
        """Add Volume Weighted Average Price"""
        try:
            if 'Volume' in df.columns:
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
                
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
                df[f'Volume_SMA_{period}'] = ta.sma(df['Volume'], length=period)
                
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
