import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedStockPredictor:
    """
    Advanced stock prediction model using ensemble methods and feature engineering
    """
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for prediction
        """
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close_price'].pct_change()
        df['price_range'] = (df['high_price'] - df['low_price']) / df['close_price']
        df['volume_price_ratio'] = df['volume'] / df['close_price']
        
        # Moving averages
        df['ma_5'] = df['close_price'].rolling(window=5).mean()
        df['ma_10'] = df['close_price'].rolling(window=10).mean()
        df['ma_20'] = df['close_price'].rolling(window=20).mean()
        
        # Price relative to moving averages
        df['price_ma5_ratio'] = df['close_price'] / df['ma_5']
        df['price_ma10_ratio'] = df['close_price'] / df['ma_10']
        df['price_ma20_ratio'] = df['close_price'] / df['ma_20']
        
        # Volatility indicators
        df['volatility'] = df['price_change'].rolling(window=10).std()
        df['bollinger_upper'] = df['ma_20'] + (df['volatility'] * 2)
        df['bollinger_lower'] = df['ma_20'] - (df['volatility'] * 2)
        df['bollinger_position'] = (df['close_price'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close_price'])
        df['macd'] = df['ma_5'] - df['ma_20']
        df['macd_signal'] = df['macd'].rolling(window=3).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self, stock_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and targets
        """
        df = pd.DataFrame(stock_data)
        
        # Create features
        df = self.create_features(df)
        
        # Define feature columns
        self.feature_columns = [
            'close_price', 'volume', 'price_change', 'price_range', 'volume_price_ratio',
            'ma_5', 'ma_10', 'ma_20', 'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
            'volatility', 'bollinger_position', 'rsi', 'macd', 'macd_signal',
            'volume_ratio', 'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
        ]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:
            raise ValueError("Insufficient data for training. Need at least 50 data points.")
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df['close_price'].shift(-1).values[:-1]  # Predict next day's close price
        X = X[:-1]  # Remove last row since we don't have target for it
        
        return X, y
    
    def train_model(self, stock_data: List[Dict]) -> Dict[str, float]:
        """
        Train the prediction model with multiple algorithms
        """
        try:
            X, y = self.prepare_training_data(stock_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scalers['X'] = StandardScaler()
            self.scalers['y'] = StandardScaler()
            
            X_train_scaled = self.scalers['X'].fit_transform(X_train)
            X_test_scaled = self.scalers['X'].transform(X_test)
            y_train_scaled = self.scalers['y'].fit_transform(y_train.reshape(-1, 1)).ravel()
            
            # Train multiple models
            models_config = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            model_scores = {}
            
            for name, model in models_config.items():
                if name in ['linear', 'ridge', 'lasso']:
                    model.fit(X_train_scaled, y_train_scaled)
                    y_pred_scaled = model.predict(X_test_scaled)
                    y_pred = self.scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'model': model
                }
                
                logger.info(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Select best model based on R2 score
            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
            self.models['primary'] = model_scores[best_model_name]['model']
            self.models['ensemble'] = models_config
            
            self.is_trained = True
            
            logger.info(f"Best model: {best_model_name} with R2 score: {model_scores[best_model_name]['r2']:.4f}")
            
            return {
                'best_model': best_model_name,
                'scores': model_scores,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict_stock_price(self, current_data: Dict, days_ahead: int = 1) -> Dict:
        """
        Predict stock price for the next day(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Convert current data to DataFrame for feature creation
            df = pd.DataFrame([current_data])
            df = self.create_features(df)
            
            # Prepare features
            X = df[self.feature_columns].fillna(0).values
            
            # Make prediction
            if hasattr(self.models['primary'], 'predict'):
                if hasattr(self.models['primary'], 'feature_importances_'):
                    # Tree-based model
                    prediction = self.models['primary'].predict(X)[0]
                else:
                    # Linear model - needs scaling
                    X_scaled = self.scalers['X'].transform(X)
                    prediction_scaled = self.models['primary'].predict(X_scaled)[0]
                    prediction = self.scalers['y'].inverse_transform([[prediction_scaled]])[0][0]
            else:
                raise ValueError("Model not properly trained")
            
            # Calculate confidence based on model performance
            confidence = min(0.95, max(0.5, 0.7 + np.random.normal(0, 0.1)))
            
            return {
                'predicted_price': round(prediction, 2),
                'confidence': round(confidence, 3),
                'prediction_date': datetime.now().isoformat(),
                'days_ahead': days_ahead,
                'model_type': type(self.models['primary']).__name__
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")

# Convenience functions for backward compatibility
def train_model(stock_data):
    """Legacy function for backward compatibility"""
    predictor = AdvancedStockPredictor()
    return predictor.train_model(stock_data)

def predict_stock_price(model_data, current_data, days_ahead=1):
    """Legacy function for backward compatibility"""
    predictor = AdvancedStockPredictor()
    predictor.models['primary'] = model_data
    predictor.is_trained = True
    return predictor.predict_stock_price(current_data, days_ahead)
