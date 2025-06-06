# utils/indicators.py - Technical Indicators & Feature Engineering Module

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TECHNICAL_INDICATORS, SCALING_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator with feature engineering capabilities.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or TECHNICAL_INDICATORS
        self.scalers = {}
        
    def add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        data = data.copy()
        
        try:
            # Simple Moving Averages
            for period in self.config.get('sma_periods', [20, 50, 200]):
                data[f'SMA_{period}'] = ta.trend.sma_indicator(data['Close'], window=period)
                
                # Price relative to SMA
                data[f'Close_SMA_{period}_Ratio'] = data['Close'] / data[f'SMA_{period}']
                
                # SMA slope (trend direction)
                data[f'SMA_{period}_Slope'] = data[f'SMA_{period}'].diff(5) / data[f'SMA_{period}'].shift(5)
            
            # Exponential Moving Averages
            for period in self.config.get('ema_periods', [12, 26]):
                data[f'EMA_{period}'] = ta.trend.ema_indicator(data['Close'], window=period)
                data[f'Close_EMA_{period}_Ratio'] = data['Close'] / data[f'EMA_{period}']
            
            # Golden Cross and Death Cross signals
            if 'EMA_12' in data.columns and 'EMA_26' in data.columns:
                data['EMA_Signal'] = np.where(data['EMA_12'] > data['EMA_26'], 1, -1)
                data['EMA_Cross'] = data['EMA_Signal'].diff()
            
            logger.info("Moving averages calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            
        return data
    
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators (RSI, MACD, Stochastic)."""
        data = data.copy()
        
        try:
            # RSI (Relative Strength Index)
            rsi_period = self.config.get('rsi_period', 14)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_period)
            
            # RSI signals
            data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
            data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
            data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
            
            # MACD (Moving Average Convergence Divergence)
            macd_fast = self.config.get('macd_fast', 12)
            macd_slow = self.config.get('macd_slow', 26)
            macd_signal = self.config.get('macd_signal', 9)
            
            data['MACD'] = ta.trend.macd(data['Close'], window_fast=macd_fast, window_slow=macd_slow)
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_fast=macd_fast, 
                                                      window_slow=macd_slow, window_sign=macd_signal)
            data['MACD_Histogram'] = ta.trend.macd_diff(data['Close'], window_fast=macd_fast, 
                                                       window_slow=macd_slow, window_sign=macd_signal)
            
            # MACD signals
            data['MACD_Bullish'] = ((data['MACD'] > data['MACD_Signal']) & 
                                   (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))).astype(int)
            data['MACD_Bearish'] = ((data['MACD'] < data['MACD_Signal']) & 
                                   (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))).astype(int)
            
            # Stochastic Oscillator
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Williams %R
            data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
            
            # Rate of Change (ROC)
            data['ROC'] = ta.momentum.roc(data['Close'], window=12)
            
            logger.info("Momentum indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            
        return data
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators (Bollinger Bands, ATR)."""
        data = data.copy()
        
        try:
            # Bollinger Bands
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2)
            
            data['BB_High'] = ta.volatility.bollinger_hband(data['Close'], window=bb_period, window_dev=bb_std)
            data['BB_Low'] = ta.volatility.bollinger_lband(data['Close'], window=bb_period, window_dev=bb_std)
            data['BB_Mid'] = ta.volatility.bollinger_mavg(data['Close'], window=bb_period)
            
            # Bollinger Band signals
            data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['BB_Mid']
            data['BB_Position'] = (data['Close'] - data['BB_Low']) / (data['BB_High'] - data['BB_Low'])
            data['BB_Squeeze'] = (data['BB_Width'] < data['BB_Width'].rolling(20).quantile(0.1)).astype(int)
            
            # Average True Range (ATR)
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            data['ATR_Ratio'] = data['ATR'] / data['Close']
            
            # Historical Volatility
            data['Historical_Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            logger.info("Volatility indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            
        return data
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        data = data.copy()
        
        try:
            # Volume Moving Averages
            data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
            
            # On-Balance Volume (OBV)
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            data['OBV_SMA'] = data['OBV'].rolling(20).mean()
            
            # Volume Price Trend (VPT)
            data['VPT'] = ta.volume.volume_price_trend(data['Close'], data['Volume'])
            
            # Accumulation/Distribution Line
            data['ADL'] = ta.volume.acc_dist_index(data['High'], data['Low'], data['Close'], data['Volume'])
            
            # Chaikin Money Flow
            data['CMF'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])
            
            # Force Index
            data['Force_Index'] = ta.volume.force_index(data['Close'], data['Volume'])
            
            # Volume-Weighted Average Price (VWAP)
            data['VWAP'] = (data['Close'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
            data['VWAP_Ratio'] = data['Close'] / data['VWAP']
            
            logger.info("Volume indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            
        return data
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators."""
        data = data.copy()
        
        try:
            # ADX (Average Directional Index)
            data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
            data['ADX_Pos'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])
            data['ADX_Neg'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
            
            # Parabolic SAR
            data['PSAR'] = ta.trend.psar_down(data['High'], data['Low'], data['Close'])
            data['PSAR_Signal'] = np.where(data['Close'] > data['PSAR'], 1, -1)
            
            # Commodity Channel Index (CCI)
            data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
            
            # Detrended Price Oscillator
            data['DPO'] = ta.trend.dpo(data['Close'])
            
            # Mass Index
            data['Mass_Index'] = ta.trend.mass_index(data['High'], data['Low'])
            
            # Trix
            data['Trix'] = ta.trend.trix(data['Close'])
            
            logger.info("Trend indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            
        return data
    
    def add_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add custom technical indicators."""
        data = data.copy()
        
        try:
            # Price momentum
            for period in [5, 10, 20]:
                data[f'Price_Momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            
            # Volume momentum
            data['Volume_Momentum'] = data['Volume'] / data['Volume'].shift(10) - 1
            
            # Price acceleration
            data['Price_Acceleration'] = data['Close'].diff(2) - data['Close'].diff(1).shift(1)
            
            # Volatility ratio
            data['Volatility_Ratio'] = (data['High'] - data['Low']) / data['Open']
            
            # Gap analysis
            data['Gap'] = data['Open'] - data['Close'].shift(1)
            data['Gap_Pct'] = data['Gap'] / data['Close'].shift(1) * 100
            data['Gap_Fill'] = np.where(
                (data['Gap'] > 0) & (data['Low'] <= data['Close'].shift(1)), 1,
                np.where((data['Gap'] < 0) & (data['High'] >= data['Close'].shift(1)), -1, 0)
            )
            
            # Support and Resistance levels
            data['Support_20'] = data['Low'].rolling(20).min()
            data['Resistance_20'] = data['High'].rolling(20).max()
            data['Support_Distance'] = (data['Close'] - data['Support_20']) / data['Close']
            data['Resistance_Distance'] = (data['Resistance_20'] - data['Close']) / data['Close']
            
            # Candlestick patterns (basic)
            data['Doji'] = (abs(data['Close'] - data['Open']) <= (data['High'] - data['Low']) * 0.1).astype(int)
            data['Hammer'] = ((data['Close'] > data['Open']) & 
                             ((data['Open'] - data['Low']) > 2 * (data['Close'] - data['Open'])) &
                             ((data['High'] - data['Close']) < (data['Close'] - data['Open']))).astype(int)
            
            # Market structure
            data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
            data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
            data['Higher_Low'] = ((data['Low'] > data['Low'].shift(1)) & 
                                 (data['Low'].shift(1) < data['Low'].shift(2))).astype(int)
            data['Lower_High'] = ((data['High'] < data['High'].shift(1)) & 
                                 (data['High'].shift(1) > data['High'].shift(2))).astype(int)
            
            logger.info("Custom indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
            
        return data
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        logger.info("Starting comprehensive technical indicators calculation")
        
        data = self.add_moving_averages(data)
        data = self.add_momentum_indicators(data)
        data = self.add_volatility_indicators(data)
        data = self.add_volume_indicators(data)
        data = self.add_trend_indicators(data)
        data = self.add_custom_indicators(data)
        
        logger.info(f"Technical indicators calculation complete. Shape: {data.shape}")
        return data
    
    def create_features_for_ml(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features suitable for machine learning.
        
        Args:
            data: DataFrame with calculated indicators
            target_col: Column to use as target variable
            
        Returns:
            Tuple of (features_df, target_series)
        """
        data = data.copy()
        
        # Calculate target (next day's return)
        data['Target'] = data[target_col].shift(-1) / data[target_col] - 1
        
        # Select feature columns (exclude OHLCV and other non-feature columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        nan_threshold = 0.5  # Remove columns with more than 50% NaN
        for col in feature_cols:
            if data[col].isna().sum() / len(data) > nan_threshold:
                feature_cols.remove(col)
                logger.warning(f"Removed feature {col} due to too many NaN values")
        
        features = data[feature_cols].copy()
        target = data['Target'].copy()
        
        # Forward fill and backward fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows where target is NaN
        valid_indices = ~target.isna()
        features = features[valid_indices]
        target = target[valid_indices]
        
        logger.info(f"Created {len(feature_cols)} features for ML with {len(features)} samples")
        
        return features, target
    
    def scale_features(self, features: pd.DataFrame, method: str = None, fit: bool = True) -> pd.DataFrame:
        """
        Scale features for machine learning.
        
        Args:
            features: Features DataFrame
            method: Scaling method ('minmax' or 'standard')
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled features DataFrame
        """
        method = method or SCALING_CONFIG.get('method', 'minmax')
        
        if method == 'minmax':
            scaler_class = MinMaxScaler
            scaler_params = {'feature_range': SCALING_CONFIG.get('feature_range', (0, 1))}
        elif method == 'standard':
            scaler_class = StandardScaler
            scaler_params = {}
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit or 'scaler' not in self.scalers:
            scaler = scaler_class(**scaler_params)
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            self.scalers['scaler'] = scaler
        else:
            scaler = self.scalers['scaler']
            scaled_features = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
        
        logger.info(f"Features scaled using {method} method")
        return scaled_features
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 60, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled features DataFrame
            sequence_length: Length of input sequences
            target_col: Target column name (if None, uses last column)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if target_col and target_col in data.columns:
            features = data.drop(columns=[target_col])
            targets = data[target_col]
        else:
            features = data.iloc[:, :-1]
            targets = data.iloc[:, -1]
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(targets.iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using correlation analysis."""
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate correlations
        correlations = features.corrwith(target).abs().sort_values(ascending=False)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(features, target)
        mi_df = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
        
        # Combine results
        importance_df = pd.DataFrame({
            'Feature': features.columns,
            'Correlation': correlations,
            'Mutual_Info': mi_df,
        }).sort_values('Correlation', ascending=False)
        
        logger.info("Feature importance calculated")
        return importance_df


# Utility functions
def calculate_technical_indicators(data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Quick function to calculate all technical indicators."""
    calculator = TechnicalIndicators(config)
    return calculator.calculate_all_indicators(data)

def prepare_ml_data(data: pd.DataFrame, sequence_length: int = 60, scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Quick function to prepare data for ML."""
    calculator = TechnicalIndicators()
    
    # Calculate indicators
    data_with_indicators = calculator.calculate_all_indicators(data)
    
    # Create features
    features, target = calculator.create_features_for_ml(data_with_indicators)
    
    # Scale features
    if scale:
        features = calculator.scale_features(features)
    
    # Add target back for sequence creation
    ml_data = features.copy()
    ml_data['Target'] = target
    
    # Create sequences
    X, y = calculator.create_sequences(ml_data, sequence_length)
    
    return X, y


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_loader import get_stock_data
    
    try:
        print("Testing technical indicators calculation...")
        
        # Get sample data
        data = get_stock_data('AAPL', period='6mo')
        print(f"Original data shape: {data.shape}")
        
        # Calculate indicators
        calculator = TechnicalIndicators()
        data_with_indicators = calculator.calculate_all_indicators(data)
        print(f"Data with indicators shape: {data_with_indicators.shape}")
        print(f"New columns added: {data_with_indicators.shape[1] - data.shape[1]}")
        
        # Create ML features
        features, target = calculator.create_features_for_ml(data_with_indicators)
        print(f"ML features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        
        # Scale features
        scaled_features = calculator.scale_features(features)
        print(f"Scaled features range: {scaled_features.min().min():.3f} to {scaled_features.max().max():.3f}")
        
        # Create sequences
        ml_data = scaled_features.copy()
        ml_data['Target'] = target
        X, y = calculator.create_sequences(ml_data, sequence_length=30)
        print(f"Sequences shape: X={X.shape}, y={y.shape}")
        
        # Feature importance
        importance = calculator.get_feature_importance(features, target)
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        print("\nTechnical indicators calculation test completed successfully!")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()