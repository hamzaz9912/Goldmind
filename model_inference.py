import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from indicators import TechnicalIndicators

# RL imports
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable-baselines3 not available. RL models will not work.")

# LSTM imports
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not work.")

class ModelInference:
    """Handle ML and RL model inference for signal generation"""

    def __init__(self, model_type="ML", uploaded_model_path=None):
        self.models = {}
        self.scalers = {}
        self.rl_models = {}
        self.lstm_models = {}
        self.models_dir = 'models'
        self.confidence_threshold = 0.7  # 70% confidence threshold
        self.model_type = model_type
        self.uploaded_model_path = uploaded_model_path
        self.accuracy_history = {}  # Track forecast accuracy
        self.penalty_factor = 0.1  # Penalty for inaccurate forecasts
    
    def load_model(self, symbol):
        """
        Load trained model and scaler for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (model, scaler) or (None, None) if not found
        """
        try:
            if self.model_type == "RL":
                return self.load_rl_model(symbol)
            elif self.model_type == "LSTM":
                return self.load_lstm_model(symbol)
            else:
                model_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}.pkl"
                scaler_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}_scaler.pkl"

                if os.path.exists(model_filename) and os.path.exists(scaler_filename):
                    if symbol not in self.models:
                        self.models[symbol] = joblib.load(model_filename)
                        self.scalers[symbol] = joblib.load(scaler_filename)

                    return self.models[symbol], self.scalers[symbol]
                else:
                    # If model doesn't exist, create a simple backup model
                    return self._create_backup_model(), self._create_backup_scaler()

        except Exception as e:
            print(f"Error loading model for {symbol}: {str(e)}")
            return None, None

    def load_rl_model(self, symbol):
        """
        Load RL model for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (rl_model, None) or (None, None) if not found
        """
        try:
            if not RL_AVAILABLE:
                print("RL not available, falling back to ML")
                return self._create_backup_model(), self._create_backup_scaler()

            # Check for uploaded model first
            if self.uploaded_model_path and os.path.exists(self.uploaded_model_path):
                if 'uploaded' not in self.rl_models:
                    try:
                        self.rl_models['uploaded'] = PPO.load(self.uploaded_model_path)
                    except:
                        self.rl_models['uploaded'] = A2C.load(self.uploaded_model_path)
                return self.rl_models['uploaded'], None

            # Try to load symbol-specific RL model
            rl_filename = f"{self.models_dir}/rl_{symbol.replace('=', '_').replace('-', '_')}.zip"
            if os.path.exists(rl_filename):
                if symbol not in self.rl_models:
                    try:
                        self.rl_models[symbol] = PPO.load(rl_filename)
                    except:
                        self.rl_models[symbol] = A2C.load(rl_filename)
                return self.rl_models[symbol], None
            else:
                # Create backup RL model
                return self._create_backup_rl_model(), None

        except Exception as e:
            print(f"Error loading RL model for {symbol}: {str(e)}")
            return self._create_backup_rl_model(), None

    def load_lstm_model(self, symbol):
        """
        Load LSTM model for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (lstm_model, scaler) or (None, None) if not found
        """
        try:
            if not LSTM_AVAILABLE:
                print("LSTM not available, falling back to ML")
                return self._create_backup_model(), self._create_backup_scaler()

            # Try to load symbol-specific LSTM model
            lstm_filename = f"{self.models_dir}/lstm_{symbol.replace('=', '_').replace('-', '_')}.h5"
            scaler_filename = f"{self.models_dir}/lstm_{symbol.replace('=', '_').replace('-', '_')}_scaler.pkl"

            if os.path.exists(lstm_filename) and os.path.exists(scaler_filename):
                if symbol not in self.lstm_models:
                    self.lstm_models[symbol] = load_model(lstm_filename)
                    self.scalers[symbol] = joblib.load(scaler_filename)
                return self.lstm_models[symbol], self.scalers[symbol]
            else:
                # Create backup LSTM model
                return self._create_backup_lstm_model(), self._create_backup_scaler()

        except Exception as e:
            print(f"Error loading LSTM model for {symbol}: {str(e)}")
            return self._create_backup_lstm_model(), self._create_backup_scaler()
    
    def _create_backup_model(self):
        """Create a simple backup model when trained model is not available"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple model with basic parameters
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data to fit the model
        X_dummy = np.random.random((100, 10))
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        
        return model
    
    def _create_backup_scaler(self):
        """Create a backup scaler"""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        # Fit with dummy data
        X_dummy = np.random.random((100, 10))
        scaler.fit(X_dummy)

        return scaler

    def _create_backup_rl_model(self):
        """Create a backup RL model"""
        if not RL_AVAILABLE:
            return self._create_backup_model()

        # Create a simple RL model with dummy environment
        from gym import spaces
        import gym

        class DummyTradingEnv(gym.Env):
            def __init__(self):
                self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

            def reset(self):
                return np.random.random(10)

            def step(self, action):
                reward = np.random.random() - 0.5
                done = np.random.random() > 0.95
                return np.random.random(10), reward, done, {}

        env = DummyVecEnv([lambda: DummyTradingEnv()])
        model = PPO('MlpPolicy', env, verbose=0)
        # Train briefly
        model.learn(total_timesteps=100)
        return model

    def _create_backup_lstm_model(self):
        """Create a backup LSTM model"""
        if not LSTM_AVAILABLE:
            return self._create_backup_model()

        # Create a simple LSTM model
        model = Sequential([
            LSTM(50, input_shape=(10, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Create dummy training data
        X_dummy = np.random.random((100, 10, 1))
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=0)

        return model
    
    def prepare_features_for_inference(self, df):
        """
        Prepare features for model inference (same as training)
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Feature array for the latest candle
        """
        try:
            features = df.copy()
            
            # Price-based features
            features['Price_Change'] = features['Close'].pct_change()
            features['High_Low_Ratio'] = features['High'] / features['Low']
            features['Close_Open_Ratio'] = features['Close'] / features['Open']
            
            # Volatility features
            features['Price_Range'] = (features['High'] - features['Low']) / features['Close']
            features['Body_Size'] = abs(features['Close'] - features['Open']) / features['Close']
            
            # Technical indicator features
            if 'EMA_20' in features.columns and 'EMA_50' in features.columns:
                features['EMA_Ratio'] = features['EMA_20'] / features['EMA_50']
                features['Price_EMA20_Ratio'] = features['Close'] / features['EMA_20']
                features['Price_EMA50_Ratio'] = features['Close'] / features['EMA_50']
            
            # RSI features
            if 'RSI_14' in features.columns:
                features['RSI_Normalized'] = (features['RSI_14'] - 50) / 50
                features['RSI_Change'] = features['RSI_14'].diff()
            
            # MACD features
            if 'MACD_12_26_9' in features.columns:
                features['MACD_Normalized'] = features['MACD_12_26_9'] / features['Close']
                if 'MACDs_12_26_9' in features.columns:
                    features['MACD_Signal_Diff'] = features['MACD_12_26_9'] - features['MACDs_12_26_9']
            
            # ATR features
            if 'ATR_14' in features.columns:
                features['ATR_Ratio'] = features['ATR_14'] / features['Close']
            
            # Volume features (if available)
            if 'Volume' in features.columns:
                features['Volume_Change'] = features['Volume'].pct_change()
                if 'Volume_SMA_20' in features.columns:
                    features['Volume_Ratio'] = features['Volume'] / features['Volume_SMA_20']
            
            # Trend features
            if 'Trend_Strength' in features.columns:
                features['Trend_Change'] = features['Trend_Strength'].diff()
            
            if 'Momentum_Score' in features.columns:
                features['Momentum_Change'] = features['Momentum_Score'].diff()
            
            # Select only numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_cols]
            
            # Handle infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            
            # Return the latest row as features
            if len(features) > 0:
                latest_features = features.iloc[-1].values.reshape(1, -1)
                
                # Ensure we have the right number of features (pad or truncate if needed)
                if latest_features.shape[1] < 10:
                    # Pad with zeros if we have fewer features
                    padding = np.zeros((1, 10 - latest_features.shape[1]))
                    latest_features = np.hstack([latest_features, padding])
                elif latest_features.shape[1] > 10:
                    # Truncate if we have more features
                    latest_features = latest_features[:, :10]
                
                return latest_features
            else:
                return np.zeros((1, 10))
                
        except Exception as e:
            print(f"Error preparing features for inference: {str(e)}")
            return np.zeros((1, 10))
    
    def calculate_stop_loss_take_profit(self, df, signal, entry_price):
        """
        Calculate stop loss and take profit levels using ATR
        
        Args:
            df: DataFrame with indicators including ATR
            signal: 'BUY' or 'SELL'
            entry_price: Entry price
            
        Returns:
            Dict with stop_loss and take_profit levels
        """
        try:
            if 'ATR_14' not in df.columns or len(df) == 0:
                # Fallback ATR calculation
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                if pd.isna(atr):
                    atr = entry_price * 0.01  # 1% fallback
            else:
                atr = df['ATR_14'].iloc[-1]
                if pd.isna(atr):
                    atr = entry_price * 0.01
            
            if signal == 'BUY':
                stop_loss = entry_price - atr
                take_profit = entry_price + (2 * atr)  # 1:2 risk-reward
            else:  # SELL
                stop_loss = entry_price + atr
                take_profit = entry_price - (2 * atr)  # 1:2 risk-reward
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr_value': atr
            }
            
        except Exception as e:
            print(f"Error calculating SL/TP: {str(e)}")
            # Fallback calculation
            if signal == 'BUY':
                return {
                    'stop_loss': entry_price * 0.99,
                    'take_profit': entry_price * 1.02,
                    'atr_value': entry_price * 0.01
                }
            else:
                return {
                    'stop_loss': entry_price * 1.01,
                    'take_profit': entry_price * 0.98,
                    'atr_value': entry_price * 0.01
                }
    
    def generate_signal(self, df, symbol):
        """
        Generate trading signal for the latest candle

        Args:
            df: DataFrame with OHLCV and indicators
            symbol: Trading symbol

        Returns:
            Dict with signal information
        """
        try:
            if df is None or len(df) < 2:
                return self._no_trade_signal()

            # Load model
            model, scaler = self.load_model(symbol)
            if model is None:
                return self._generate_technical_signal(df)

            # Get entry price (latest close)
            entry_price = df['Close'].iloc[-1]

            if self.model_type == "RL":
                # RL model prediction
                features = self.prepare_features_for_inference(df)
                if features is None:
                    return self._generate_technical_signal(df)

                # RL models expect unscaled features often
                try:
                    if hasattr(model, 'predict'):
                        # Check if it's a real RL model or fallback ML model
                        if hasattr(model, 'policy'):  # Real RL model
                            action, _ = model.predict(features, deterministic=True)
                        else:  # Fallback ML model
                            features_scaled = scaler.transform(features) if scaler else features
                            action = model.predict(features_scaled)[0]
                            action = 2 if action == 1 else 0  # Convert ML prediction to RL action

                        # Assume action 0 = SELL, 1 = HOLD, 2 = BUY
                        if action == 2:
                            signal = 'BUY'
                            confidence = 75.0
                        elif action == 0:
                            signal = 'SELL'
                            confidence = 75.0
                        else:
                            signal = 'NO TRADE'
                            confidence = 50.0
                    else:
                        return self._generate_technical_signal(df)
                except Exception as e:
                    print(f"RL prediction error: {e}")
                    return self._generate_technical_signal(df)
            elif self.model_type == "LSTM":
                # LSTM model prediction
                features = self.prepare_features_for_inference(df)
                if features is None:
                    return self._generate_technical_signal(df)

                # LSTM expects 3D input (samples, timesteps, features)
                if scaler is None:
                    return self._generate_technical_signal(df)

                try:
                    features_scaled = scaler.transform(features)
                    # Reshape for LSTM (assuming sequence length of 10)
                    seq_length = 10
                    if features_scaled.shape[1] >= seq_length:
                        X_lstm = features_scaled[:, -seq_length:].reshape(1, seq_length, -1)
                    else:
                        # Pad if necessary
                        padding = np.zeros((1, seq_length - features_scaled.shape[1], features_scaled.shape[1]))
                        X_lstm = np.concatenate([padding, features_scaled.reshape(1, features_scaled.shape[1], -1)], axis=1)

                    if hasattr(model, 'predict'):
                        # Check if it's a real LSTM model or fallback ML model
                        if hasattr(model, 'layers'):  # Real LSTM model
                            prediction_prob = model.predict(X_lstm, verbose=0)[0][0]
                        else:  # Fallback ML model
                            prediction_prob = model.predict_proba(features_scaled)[0][1]  # Probability of positive class

                        prediction = 1 if prediction_prob > 0.5 else 0
                        confidence = max(prediction_prob, 1 - prediction_prob) * 100
                    else:
                        return self._generate_technical_signal(df)

                except Exception as e:
                    print(f"LSTM prediction error: {e}")
                    return self._generate_technical_signal(df)
            else:
                # ML model prediction
                if scaler is None:
                    return self._generate_technical_signal(df)

                # Prepare features
                features = self.prepare_features_for_inference(df)

                if features is None:
                    return self._generate_technical_signal(df)

                # Scale features
                try:
                    features_scaled = scaler.transform(features)
                except:
                    # If scaling fails, use technical signal
                    return self._generate_technical_signal(df)

                # Get prediction and probability
                prediction = model.predict(features_scaled)[0]
                try:
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = max(probabilities) * 100
                except:
                    # Some models might not have predict_proba
                    confidence = 60.0  # Default confidence

                # Determine signal based on prediction and confidence
                if prediction == 1 and confidence >= self.confidence_threshold * 100:
                    signal = 'BUY'
                elif prediction == 0 and confidence >= self.confidence_threshold * 100:
                    signal = 'SELL'
                else:
                    signal = 'NO TRADE'

            # Apply penalty for inaccurate forecasts
            if symbol in self.accuracy_history:
                accuracy_penalty = self.accuracy_history[symbol].get('penalty', 0)
                confidence = max(10, confidence - accuracy_penalty)  # Minimum 10% confidence

            # Determine signal based on prediction and confidence
            if self.model_type in ["RL", "LSTM"]:
                if prediction == 1 and confidence >= self.confidence_threshold * 100:
                    signal = 'BUY'
                elif prediction == 0 and confidence >= self.confidence_threshold * 100:
                    signal = 'SELL'
                else:
                    signal = 'NO TRADE'
            # For ML models, signal is already determined above

            # Calculate stop loss and take profit
            if signal != 'NO TRADE':
                sl_tp = self.calculate_stop_loss_take_profit(df, signal, entry_price)
            else:
                sl_tp = {'stop_loss': entry_price, 'take_profit': entry_price, 'atr_value': 0}

            return {
                'signal': signal,
                'confidence': confidence,
                'entry': entry_price,
                'stop_loss': sl_tp['stop_loss'],
                'take_profit': sl_tp['take_profit'],
                'atr_value': sl_tp['atr_value'],
                'timestamp': datetime.now(),
                'model_prediction': self.model_type if self.model_type in ["RL", "LSTM"] else prediction,
                'symbol': symbol,
                'model_type': self.model_type
            }

        except Exception as e:
            print(f"Error generating signal for {symbol}: {str(e)}")
            return self._generate_technical_signal(df)
    
    def _generate_technical_signal(self, df):
        """
        Generate signal using technical indicators only (fallback)
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dict with signal information
        """
        try:
            if len(df) < 2:
                return self._no_trade_signal()
            
            latest = df.iloc[-1]
            entry_price = latest['Close']
            
            # Simple technical signal logic
            signals = []
            
            # EMA signal
            if 'EMA_20' in latest and 'EMA_50' in latest:
                if latest['EMA_20'] > latest['EMA_50']:
                    signals.append(1)
                else:
                    signals.append(-1)
            
            # RSI signal
            if 'RSI_14' in latest:
                if latest['RSI_14'] < 30:
                    signals.append(1)  # Oversold - buy
                elif latest['RSI_14'] > 70:
                    signals.append(-1)  # Overbought - sell
                else:
                    signals.append(0)  # Neutral
            
            # MACD signal
            if 'MACD_12_26_9' in latest and 'MACDs_12_26_9' in latest:
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    signals.append(1)
                else:
                    signals.append(-1)
            
            # Calculate average signal
            if signals:
                avg_signal = np.mean(signals)
                
                if avg_signal > 0.3:
                    signal = 'BUY'
                    confidence = min(80.0, 50 + abs(avg_signal) * 30)
                elif avg_signal < -0.3:
                    signal = 'SELL'
                    confidence = min(80.0, 50 + abs(avg_signal) * 30)
                else:
                    signal = 'NO TRADE'
                    confidence = 50.0
            else:
                signal = 'NO TRADE'
                confidence = 50.0
            
            # Calculate SL/TP
            if signal != 'NO TRADE':
                sl_tp = self.calculate_stop_loss_take_profit(df, signal, entry_price)
            else:
                sl_tp = {'stop_loss': entry_price, 'take_profit': entry_price, 'atr_value': 0}
            
            return {
                'signal': signal,
                'confidence': confidence,
                'entry': entry_price,
                'stop_loss': sl_tp['stop_loss'],
                'take_profit': sl_tp['take_profit'],
                'atr_value': sl_tp['atr_value'],
                'timestamp': datetime.now(),
                'model_prediction': 'Technical',
                'symbol': 'Unknown'
            }
            
        except Exception as e:
            print(f"Error generating technical signal: {str(e)}")
            return self._no_trade_signal()
    
    def _no_trade_signal(self):
        """Return a no-trade signal"""
        return {
            'signal': 'NO TRADE',
            'confidence': 0.0,
            'entry': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'atr_value': 0.0,
            'timestamp': datetime.now(),
            'model_prediction': 'None',
            'symbol': 'Unknown'
        }
    
    def batch_generate_signals(self, data_dict):
        """
        Generate signals for multiple assets
        
        Args:
            data_dict: Dict with {symbol: dataframe} pairs
            
        Returns:
            Dict with {symbol: signal_info} pairs
        """
        signals = {}
        
        for symbol, df in data_dict.items():
            try:
                signal = self.generate_signal(df, symbol)
                signals[symbol] = signal
            except Exception as e:
                print(f"Error generating signal for {symbol}: {str(e)}")
                signals[symbol] = self._no_trade_signal()
        
        return signals
    
    def get_model_info(self, symbol):
        """
        Get information about the trained model

        Args:
            symbol: Trading symbol

        Returns:
            Dict with model information
        """
        try:
            info_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}_info.json"

            if os.path.exists(info_filename):
                import json
                with open(info_filename, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'symbol': symbol,
                    'model_type': 'Technical Fallback',
                    'accuracy': 'N/A',
                    'training_date': 'N/A'
                }

        except Exception as e:
            print(f"Error getting model info for {symbol}: {str(e)}")
            return {'error': str(e)}

    def update_forecast_accuracy(self, symbol, forecasted_price, actual_price, forecast_timestamp):
        """
        Update forecast accuracy tracking and apply penalties

        Args:
            symbol: Trading symbol
            forecasted_price: Price predicted by the model
            actual_price: Actual market price
            forecast_timestamp: When the forecast was made
        """
        try:
            if symbol not in self.accuracy_history:
                self.accuracy_history[symbol] = {
                    'forecasts': [],
                    'accuracy_score': 1.0,
                    'penalty': 0.0
                }

            # Calculate forecast error
            error = abs(forecasted_price - actual_price) / actual_price
            accuracy = max(0, 1 - error)  # 1.0 = perfect, 0.0 = completely wrong

            # Store forecast
            self.accuracy_history[symbol]['forecasts'].append({
                'timestamp': forecast_timestamp,
                'forecasted': forecasted_price,
                'actual': actual_price,
                'error': error,
                'accuracy': accuracy
            })

            # Keep only last 20 forecasts
            if len(self.accuracy_history[symbol]['forecasts']) > 20:
                self.accuracy_history[symbol]['forecasts'] = self.accuracy_history[symbol]['forecasts'][-20:]

            # Update overall accuracy score (weighted average of recent forecasts)
            recent_forecasts = self.accuracy_history[symbol]['forecasts'][-10:]  # Last 10 forecasts
            if recent_forecasts:
                weights = np.linspace(0.5, 1.0, len(recent_forecasts))  # More recent forecasts have higher weight
                weighted_accuracy = np.average([f['accuracy'] for f in recent_forecasts], weights=weights)
                self.accuracy_history[symbol]['accuracy_score'] = weighted_accuracy

                # Apply penalty based on accuracy
                if weighted_accuracy < 0.7:  # Less than 70% accuracy
                    penalty_increase = self.penalty_factor * (0.7 - weighted_accuracy) / 0.7
                    self.accuracy_history[symbol]['penalty'] = min(50, self.accuracy_history[symbol]['penalty'] + penalty_increase)
                else:
                    # Gradually reduce penalty for good performance
                    self.accuracy_history[symbol]['penalty'] = max(0, self.accuracy_history[symbol]['penalty'] - 0.5)

        except Exception as e:
            print(f"Error updating forecast accuracy for {symbol}: {str(e)}")

    def get_forecast_accuracy(self, symbol):
        """
        Get current forecast accuracy for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dict with accuracy information
        """
        if symbol in self.accuracy_history:
            return {
                'accuracy_score': self.accuracy_history[symbol]['accuracy_score'],
                'penalty': self.accuracy_history[symbol]['penalty'],
                'forecast_count': len(self.accuracy_history[symbol]['forecasts'])
            }
        else:
            return {
                'accuracy_score': 1.0,
                'penalty': 0.0,
                'forecast_count': 0
            }
