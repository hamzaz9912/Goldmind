"""
Advanced AutoML Module using XGBoost and LightGBM
"""
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

from model_pipeline import ModelPipeline

class AutoMLPipeline(ModelPipeline):
    """
    Advanced AutoML pipeline with XGBoost and LightGBM
    """
    
    def __init__(self, models_dir: str = "models"):
        super().__init__(models_dir)
        self.automl_models_dir = self.models_dir / "automl"
        self.automl_models_dir.mkdir(exist_ok=True)
    
    def get_xgboost_model(self):
        """Get XGBoost classifier with optimized parameters"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost'):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'xgboost' or 'lightgbm'
            
        Returns:
            Best estimator
        """
        print(f"Performing hyperparameter tuning for {model_type}...")
        
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Use TimeSeriesSplit for time-aware cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def train_ensemble_models(self, symbol: str, interval: str = '30m', 
                             periods: int = 500, test_size: float = 0.3,
                             tune_hyperparameters: bool = False) -> Dict:
        """
        Train ensemble of advanced models (XGBoost, LightGBM, RF, GB)
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            periods: Number of periods to fetch
            test_size: Proportion of data for testing
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training and evaluation results
        """
        print(f"\n{'='*60}")
        print(f"AutoML Training: {symbol}")
        print(f"{'='*60}")
        
        # Fetch and prepare data
        df = self.data_fetcher.get_ohlcv_data(symbol, interval, periods)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return None
        
        # CRITICAL: Prepare features FIRST (before creating target)
        # This ensures features are calculated only on available data
        df_features, feature_cols = self.prepare_features(df)
        
        # Create target on the original data
        df_with_target = self.create_target(df, lookahead=5, threshold=0.002)
        
        # CRITICAL: Shift features by lookahead to prevent data leakage
        # Features at time t should only use information available at time t
        # to predict target at time t+lookahead
        lookahead = 5
        df_features_shifted = df_features.shift(lookahead).dropna()
        
        # Align features and target (only keep rows that exist in both)
        common_index = df_features_shifted.index.intersection(df_with_target.index)
        df_features_final = df_features_shifted.loc[common_index]
        df_target_final = df_with_target.loc[common_index]
        
        X = df_features_final[feature_cols].values
        y = df_target_final['target'].values
        
        # Store test dataframe for backtesting (use the aligned target df)
        df_test_full = df_target_final
        
        # Time series split for training/testing
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df_test_full.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {}
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        models['rf'] = rf_model
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        models['gb'] = gb_model
        
        # XGBoost
        if tune_hyperparameters:
            print("Training XGBoost with hyperparameter tuning...")
            xgb_model = self.hyperparameter_tuning(X_train_scaled, y_train, 'xgboost')
        else:
            print("Training XGBoost...")
            xgb_model = self.get_xgboost_model()
            xgb_model.fit(X_train_scaled, y_train)
        models['xgb'] = xgb_model
        
        
        # Evaluate all models
        print(f"\n{'='*60}")
        print("Model Evaluation Results")
        print(f"{'='*60}")
        
        results = {}
        best_accuracy = 0
        best_model_name = None
        
        for name, model in models.items():
            pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, zero_division=0)
            recall = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            print(f"\n{name.upper()} Model:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
            
            # Backtest
            backtest_results = self.backtest_strategy(df_test, pred, initial_capital=10000)
            
            print(f"Backtest - Return: {backtest_results['total_return']:.2f}%, "
                  f"Sharpe: {backtest_results['sharpe_ratio']:.2f}, "
                  f"Win Rate: {backtest_results['win_rate']:.2f}%")
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'backtest': backtest_results
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        
        # Ensemble prediction (voting)
        print(f"\n{'='*60}")
        print("Ensemble Model (Voting)")
        print(f"{'='*60}")
        
        ensemble_predictions = []
        for model in models.values():
            ensemble_predictions.append(model.predict(X_test_scaled))
        
        # Majority voting
        ensemble_pred = np.round(np.mean(ensemble_predictions, axis=0)).astype(int)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")
        
        ensemble_backtest = self.backtest_strategy(df_test, ensemble_pred, initial_capital=10000)
        print(f"Ensemble Backtest - Return: {ensemble_backtest['total_return']:.2f}%, "
              f"Sharpe: {ensemble_backtest['sharpe_ratio']:.2f}, "
              f"Win Rate: {ensemble_backtest['win_rate']:.2f}%")
        
        # Save best model
        print(f"\nBest single model: {best_model_name.upper()} with accuracy {best_accuracy:.3f}")
        
        model_name = symbol.replace("-", "_").replace("=", "_")
        
        # Save best model
        best_model = results[best_model_name]['model']
        model_path = self.automl_models_dir / f"{model_name}_best.pkl"
        scaler_path = self.automl_models_dir / f"{model_name}_scaler.pkl"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save ensemble models
        ensemble_path = self.automl_models_dir / f"{model_name}_ensemble.pkl"
        joblib.dump(models, ensemble_path)
        
        # Save metadata
        model_info = {
            'symbol': symbol,
            'interval': interval,
            'trained_date': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'feature_names': feature_cols,
            'models_performance': {
                name: {
                    'accuracy': res['accuracy'],
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'f1': res['f1'],
                    'backtest_return': res['backtest']['total_return'],
                    'backtest_sharpe': res['backtest']['sharpe_ratio'],
                    'win_rate': res['backtest']['win_rate']
                }
                for name, res in results.items()
            },
            'ensemble_backtest': {
                'total_return': ensemble_backtest['total_return'],
                'sharpe_ratio': ensemble_backtest['sharpe_ratio'],
                'win_rate': ensemble_backtest['win_rate']
            }
        }
        
        info_path = self.automl_models_dir / f"{model_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\nModels saved:")
        print(f"  Best model: {model_path}")
        print(f"  Ensemble: {ensemble_path}")
        print(f"  Info: {info_path}")
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scaler': scaler,
            'results': results,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_backtest': ensemble_backtest,
            'feature_names': feature_cols
        }
    
    def batch_train_automl(self, symbols: List[str], interval: str = '30m', 
                          periods: int = 500, tune_hyperparameters: bool = False):
        """
        Train AutoML models for multiple symbols
        
        Args:
            symbols: List of trading symbols
            interval: Data interval
            periods: Number of periods to fetch
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.train_ensemble_models(symbol, interval, periods, 
                                                   tune_hyperparameters=tune_hyperparameters)
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
                    'best_model': result['best_model_name'],
                    'ensemble_accuracy': result['ensemble_accuracy'],
                    'ensemble_return': result['ensemble_backtest']['total_return'],
                    'ensemble_sharpe': result['ensemble_backtest']['sharpe_ratio'],
                    'ensemble_win_rate': result['ensemble_backtest']['win_rate']
                }
                for symbol, result in results.items()
            }
        }
        
        summary_path = self.automl_models_dir / "automl_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"AutoML Batch Training Complete")
        print(f"{'='*60}")
        print(f"Models trained: {len(results)}")
        print(f"Summary saved: {summary_path}")
        
        return results

if __name__ == "__main__":
    # Example usage
    automl = AutoMLPipeline()
    
    # Train AutoML models for top assets
    symbols = ['BTC-USD', 'ETH-USD', 'GC=F']
    
    automl.batch_train_automl(symbols, interval='30m', periods=500, tune_hyperparameters=False)
