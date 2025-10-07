import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetch import DataFetcher
from indicators import TechnicalIndicators

class ModelTrainer:
    """Train and save ML models for signal generation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.models_dir = 'models'
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """
        Prepare features for ML model training
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Feature DataFrame
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
            
            # Select only numeric columns for features
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_cols]
            
            # Remove any infinite or NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return df
    
    def create_target_variable(self, df, lookforward=1, threshold=0.001):
        """
        Create target variable for classification
        
        Args:
            df: DataFrame with price data
            lookforward: Number of periods to look forward
            threshold: Minimum price change threshold
            
        Returns:
            Target array (1=Buy, 0=Sell)
        """
        try:
            # Calculate future returns
            future_close = df['Close'].shift(-lookforward)
            current_close = df['Close']
            
            future_return = (future_close - current_close) / current_close
            
            # Create binary target based on threshold
            # 1 = Buy (price will go up), 0 = Sell (price will go down)
            target = np.where(future_return > threshold, 1, 0)
            
            return target
            
        except Exception as e:
            print(f"Error creating target variable: {str(e)}")
            return np.zeros(len(df))
    
    def train_model(self, symbol, timeframe='30m', periods=1000):
        """
        Train ML model for a specific symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Time interval
            periods: Number of periods to fetch for training
            
        Returns:
            Trained model and performance metrics
        """
        try:
            print(f"Training model for {symbol}...")
            
            # Fetch training data
            fetcher = DataFetcher()
            df = fetcher.get_ohlcv_data(symbol, timeframe, periods)
            
            if df is None or len(df) < 100:
                print(f"Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators
            indicators = TechnicalIndicators()
            df_with_indicators = indicators.calculate_all_indicators(df)
            
            # Prepare features
            features_df = self.prepare_features(df_with_indicators)
            
            # Create target variable
            target = self.create_target_variable(df_with_indicators)
            
            # Remove last row (no future data for target)
            features_df = features_df.iloc[:-1]
            target = target[:-1]
            
            # Remove rows with NaN values
            valid_indices = ~(features_df.isnull().any(axis=1) | np.isnan(target))
            features_df = features_df[valid_indices]
            target = target[valid_indices]
            
            if len(features_df) < 50:
                print(f"Insufficient valid data for {symbol}")
                return None
            
            # Store feature columns
            self.feature_columns = features_df.columns.tolist()
            
            # Split data (time-aware split, NO shuffling for time series)
            split_idx = int(len(features_df) * 0.8)
            X_train = features_df.iloc[:split_idx]
            X_test = features_df.iloc[split_idx:]
            y_train = target[:split_idx]
            y_test = target[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Train Gradient Boosting model
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_score = rf_model.score(X_test_scaled, y_test)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            print(f"Random Forest accuracy: {rf_score:.3f}")
            print(f"Gradient Boosting accuracy: {gb_score:.3f}")
            
            # Choose best model
            if rf_score >= gb_score:
                best_model = rf_model
                model_type = 'RandomForest'
                accuracy = rf_score
            else:
                best_model = gb_model
                model_type = 'GradientBoosting'
                accuracy = gb_score
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"Cross-validation: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
            
            # Save model and scaler
            model_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}.pkl"
            scaler_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}_scaler.pkl"
            
            joblib.dump(best_model, model_filename)
            joblib.dump(scaler, scaler_filename)
            
            # Store in memory
            self.models[symbol] = best_model
            self.scalers[symbol] = scaler
            
            # Performance metrics
            y_pred = best_model.predict(X_test_scaled)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            training_info = {
                'symbol': symbol,
                'model_type': model_type,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'training_date': datetime.now().isoformat()
            }
            
            # Save training info
            import json
            info_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}_info.json"
            with open(info_filename, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"Model saved: {model_filename}")
            return training_info
            
        except Exception as e:
            print(f"Error training model for {symbol}: {str(e)}")
            return None
    
    def train_all_models(self):
        """Train models for all supported assets"""
        
        # Assets to train
        assets = [
            'XAUUSD=X',  # Gold
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
            'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD',
            'UNI-USD', 'LTC-USD', 'BCH-USD', 'ALGO-USD', 'VET-USD',
            'ICP-USD', 'ATOM-USD', 'FIL-USD', 'TRX-USD', 'ETC-USD'
        ]
        
        training_results = []
        
        for asset in assets:
            print(f"\n{'='*50}")
            print(f"Training model for {asset}")
            print(f"{'='*50}")
            
            result = self.train_model(asset)
            if result:
                training_results.append(result)
            
            # Small delay to avoid overwhelming APIs
            import time
            time.sleep(1)
        
        print(f"\n{'='*50}")
        print(f"Training completed for {len(training_results)}/{len(assets)} assets")
        print(f"{'='*50}")
        
        return training_results
    
    def update_model(self, symbol, new_data_periods=100):
        """
        Update existing model with new data
        
        Args:
            symbol: Trading symbol
            new_data_periods: Number of new periods to include
        """
        try:
            print(f"Updating model for {symbol}...")
            
            # Load existing model info
            info_filename = f"{self.models_dir}/{symbol.replace('=', '_').replace('-', '_')}_info.json"
            if os.path.exists(info_filename):
                import json
                with open(info_filename, 'r') as f:
                    old_info = json.load(f)
                print(f"Last training: {old_info.get('training_date', 'Unknown')}")
            
            # Retrain with new data
            return self.train_model(symbol, periods=new_data_periods + 500)
            
        except Exception as e:
            print(f"Error updating model for {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Train a single model
    result = trainer.train_model('BTC-USD', '30m')
    
    # Or train all models
    # results = trainer.train_all_models()
