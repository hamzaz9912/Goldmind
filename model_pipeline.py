"""
Model Training Pipeline with Backtesting and Performance Tracking
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from data_fetch import DataFetcher
from indicators import TechnicalIndicators

class ModelPipeline:
    """
    Comprehensive ML pipeline for training, backtesting, and evaluating trading models
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        
        self.performance_log = self.models_dir / "performance_log.json"
        self.backtest_results = self.models_dir / "backtest_results.json"
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature set from raw data
        
        Args:
            df: Raw price dataframe
            
        Returns:
            Tuple of (features dataframe, feature names)
        """
        df_features = df.copy()
        
        # Calculate technical indicators
        df_features = self.tech_indicators.calculate_all_indicators(df_features)
        
        # Add price-based features
        df_features['returns'] = df_features['Close'].pct_change()
        df_features['log_returns'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
        
        # Volume features
        df_features['volume_ma_5'] = df_features['Volume'].rolling(window=5).mean()
        df_features['volume_ma_10'] = df_features['Volume'].rolling(window=10).mean()
        df_features['volume_ratio'] = df_features['Volume'] / df_features['volume_ma_10']
        
        # Price momentum features
        df_features['momentum_5'] = df_features['Close'] - df_features['Close'].shift(5)
        df_features['momentum_10'] = df_features['Close'] - df_features['Close'].shift(10)
        df_features['momentum_20'] = df_features['Close'] - df_features['Close'].shift(20)
        
        # Volatility features
        df_features['volatility_5'] = df_features['returns'].rolling(window=5).std()
        df_features['volatility_10'] = df_features['returns'].rolling(window=10).std()
        df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
        
        # Drop rows with NaN values
        df_features = df_features.dropna()
        
        # Select feature columns (exclude raw OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols and col != 'target']
        
        return df_features, feature_cols
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """
        Create target variable for classification
        
        Args:
            df: Dataframe with price data
            lookahead: Number of periods to look ahead
            threshold: Price change threshold for buy signal
            
        Returns:
            Dataframe with target column
        """
        df_target = df.copy()
        
        # Calculate forward returns
        df_target['forward_return'] = df_target['Close'].shift(-lookahead) / df_target['Close'] - 1
        
        # Create binary target: 1 = Buy (price will increase), 0 = No Buy/Sell
        df_target['target'] = (df_target['forward_return'] > threshold).astype(int)
        
        # Remove lookahead rows
        df_target = df_target[:-lookahead]
        
        return df_target
    
    def backtest_strategy(self, df: pd.DataFrame, predictions: np.ndarray, 
                         initial_capital: float = 10000.0) -> Dict:
        """
        Backtest trading strategy based on model predictions
        
        Args:
            df: Historical price data
            predictions: Model predictions (0 or 1)
            initial_capital: Starting capital
            
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(len(predictions)):
            current_price = df.iloc[i]['Close']
            
            # Buy signal
            if predictions[i] == 1 and position == 0:
                shares = capital / current_price
                position = shares
                entry_price = current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'date': df.index[i]
                })
            
            # Sell signal (or exit after holding period)
            elif predictions[i] == 0 and position > 0:
                capital = position * current_price
                profit = (current_price - entry_price) * position
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'profit': profit,
                    'date': df.index[i]
                })
                position = 0
            
            # Update equity
            if position > 0:
                equity = position * current_price
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Calculate metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        equity_curve_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve_arr)
        drawdown = (equity_curve_arr - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Win rate
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        total_trades = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def train_and_backtest(self, symbol: str, interval: str = '30m', 
                          periods: int = 500, test_size: float = 0.3) -> Dict:
        """
        Complete pipeline: fetch data, train model, and backtest
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            periods: Number of periods to fetch
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training and backtest results
        """
        print(f"\n{'='*60}")
        print(f"Training and Backtesting: {symbol}")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.data_fetcher.get_ohlcv_data(symbol, interval, periods)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return None
        
        # Create target
        df = self.create_target(df, lookahead=5, threshold=0.002)
        
        # Prepare features
        df_features, feature_cols = self.prepare_features(df)
        
        # Align features and target
        df_features = df_features[df_features.index.isin(df.index)]
        df = df[df.index.isin(df_features.index)]
        
        X = df_features[feature_cols].values
        y = df['target'].values
        
        # Time series split for training/testing
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train-test split (time-aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        print(f"\nModel Performance:")
        print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"Gradient Boosting Accuracy: {gb_accuracy:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
        print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Detailed metrics
        print(f"\nClassification Report (Random Forest):")
        print(classification_report(y_test, rf_pred))
        
        # Backtest
        backtest_results = self.backtest_strategy(df_test, rf_pred, initial_capital=10000)
        
        print(f"\nBacktest Results:")
        print(f"Total Return: {backtest_results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        print(f"Total Trades: {backtest_results['total_trades']}")
        
        # Save model
        model_name = symbol.replace("-", "_").replace("=", "_")
        model_path = self.models_dir / f"{model_name}.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        
        joblib.dump(rf_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save model metadata
        model_info = {
            'symbol': symbol,
            'interval': interval,
            'trained_date': datetime.now().isoformat(),
            'accuracy': rf_accuracy,
            'cv_score': cv_scores.mean(),
            'feature_names': feature_cols,
            'backtest': {
                'total_return': backtest_results['total_return'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown'],
                'win_rate': backtest_results['win_rate'],
                'total_trades': backtest_results['total_trades']
            }
        }
        
        info_path = self.models_dir / f"{model_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\nModel saved: {model_path}")
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'accuracy': rf_accuracy,
            'cv_score': cv_scores.mean(),
            'backtest': backtest_results,
            'feature_names': feature_cols
        }
    
    def batch_train_models(self, symbols: List[str], interval: str = '30m', periods: int = 500):
        """
        Train models for multiple symbols
        
        Args:
            symbols: List of trading symbols
            interval: Data interval
            periods: Number of periods to fetch
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.train_and_backtest(symbol, interval, periods)
                if result:
                    results[symbol] = result
            except Exception as e:
                print(f"Error training {symbol}: {str(e)}")
        
        # Save batch results summary
        summary = {
            'trained_date': datetime.now().isoformat(),
            'symbols': list(results.keys()),
            'results': {
                symbol: {
                    'accuracy': result['accuracy'],
                    'cv_score': result['cv_score'],
                    'total_return': result['backtest']['total_return'],
                    'sharpe_ratio': result['backtest']['sharpe_ratio'],
                    'win_rate': result['backtest']['win_rate']
                }
                for symbol, result in results.items()
            }
        }
        
        summary_path = self.models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Batch Training Complete")
        print(f"{'='*60}")
        print(f"Models trained: {len(results)}")
        print(f"Summary saved: {summary_path}")
        
        return results

if __name__ == "__main__":
    # Example usage
    pipeline = ModelPipeline()
    
    # Train models for top assets
    symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD',
        'GC=F',  # Gold
        'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
    ]
    
    pipeline.batch_train_models(symbols, interval='30m', periods=500)
