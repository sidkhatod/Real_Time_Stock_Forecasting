# models/lstm_model.py - LSTM Model for Stock Price Forecasting

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_CONFIG, get_model_file_path

# Setup logging
logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    """
    Advanced LSTM model for stock price prediction with technical indicators.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.history = None
        self.is_trained = False
        
    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Create LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            Bidirectional(LSTM(
                units=100,
                return_sequences=True,
                input_shape=input_shape
            )),
            Dropout(self.config['dropout_rate']),
            
            # Second LSTM layer
            Bidirectional(LSTM(
                units=80,
                return_sequences=True
            )),
            Dropout(self.config['dropout_rate']),
            
            # Third LSTM layer
            LSTM(
                units=50,
                return_sequences=False
            ),
            Dropout(self.config['dropout_rate']),
            
            # Dense layers
            Dense(units=25, activation='relu'),
            Dropout(self.config['dropout_rate'] / 2),
            
            Dense(units=1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"LSTM model created with input shape: {input_shape}")
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Ensure data is sorted by index (date)
        data = data.sort_index()
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Split features and target
        if target_col in data.columns:
            features = data.drop(columns=[target_col])
            target = data[target_col].values
        else:
            features = data.iloc[:, :-1]
            target = data.iloc[:, -1].values
        
        self.feature_columns = features.columns.tolist()
        
        # Create sequences
        X, y = self._create_sequences(features.values, target)
        
        # Split data
        split_idx = int(len(X) * (1 - self.config['test_split']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        sequence_length = self.config['sequence_length']
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              ticker: str = 'STOCK') -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            ticker: Stock ticker for model saving
            
        Returns:
            Training history dictionary
        """
        # Create validation split if not provided
        if X_val is None or y_val is None:
            val_split = self.config['validation_split']
            split_idx = int(len(X_train) * (1 - val_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Create model
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=get_model_file_path(ticker, 'lstm'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting LSTM model training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        self.is_trained = True
        logger.info("LSTM model training completed")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(y_pred))
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}, Direction Accuracy: {direction_accuracy:.2f}%")
        return metrics
    
    def predict_next(self, recent_data: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Predict next N steps using recent data.
        
        Args:
            recent_data: Recent data with same features as training
            steps: Number of future steps to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare the last sequence
        sequence_length = self.config['sequence_length']
        
        if len(recent_data) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} recent data points")
        
        # Get the last sequence
        last_sequence = recent_data[self.feature_columns].tail(sequence_length).values
        last_sequence = last_sequence.reshape(1, sequence_length, -1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence (this is simplified - in practice, you'd need to update all features)
            # For now, we'll just use the prediction and assume other features remain similar
            if steps == 1:
                break
            
            # This is a simplified approach - in practice, you'd need to predict all features
            # or use a more sophisticated multi-step prediction approach
            new_row = current_sequence[0, -1:].copy()
            new_row[0, 0] = next_pred  # Update price feature
            current_sequence = np.concatenate([current_sequence[0, 1:], new_row], axis=0)
            current_sequence = current_sequence.reshape(1, sequence_length, -1)
        
        return np.array(predictions)
    
    def save_model(self, ticker: str) -> None:
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = get_model_file_path(ticker, 'lstm')
        self.model.save(model_path)
        
        # Save additional metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, ticker: str) -> None:
        """Load a pre-trained model."""
        model_path = get_model_file_path(ticker, 'lstm')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_columns = metadata.get('feature_columns', [])
            self.config.update(metadata.get('config', {}))
            self.is_trained = metadata.get('is_trained', True)
        
        logger.info(f"Model loaded from {model_path}")
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Training MAPE')
        axes[1, 0].plot(self.history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].legend()
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "LSTM Predictions vs Actual") -> None:
        """Plot predictions vs actual values."""
        plt.figure(figsize=(15, 8))
        
        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        residuals = y_true - y_pred
        plt.plot(residuals, alpha=0.7)
        plt.title('Residuals (Actual - Predicted)')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def train_lstm_model(data: pd.DataFrame, ticker: str, target_col: str = 'Close') -> LSTMStockPredictor:
    """
    Quick function to train an LSTM model.
    
    Args:
        data: DataFrame with features and target
        ticker: Stock ticker symbol
        target_col: Target column name
        
    Returns:
        Trained LSTM model
    """
    model = LSTMStockPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(data, target_col)
    
    # Train model
    history = model.train(X_train, y_train, ticker=ticker)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model(ticker)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test LSTM model
    try:
        from utils.data_loader import get_stock_data
        from utils.indicators import calculate_technical_indicators
        
        print("Testing LSTM model...")
        
        # Get and prepare data
        ticker = 'AAPL'
        data = get_stock_data(ticker, period='2y')
        data_with_indicators = calculate_technical_indicators(data)
        
        # Remove non-numeric columns and handle NaN
        numeric_data = data_with_indicators.select_dtypes(include=[np.number])
        numeric_data = numeric_data.dropna()
        
        print(f"Data shape: {numeric_data.shape}")
        
        # Train model
        model = LSTMStockPredictor()
        X_train, X_test, y_train, y_test = model.prepare_data(numeric_data, 'Close')
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Train (with fewer epochs for testing)
        test_config = LSTM_CONFIG.copy()
        test_config['epochs'] = 5
        model.config = test_config
        
        history = model.train(X_train, y_train, ticker=ticker)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test R²: {metrics['r2']:.4f}")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
        
        # Test prediction
        recent_data = numeric_data.tail(100)
        next_pred = model.predict_next(recent_data, steps=1)
        print(f"Next day prediction: ${next_pred[0]:.2f}")
        
        print("LSTM model test completed successfully!")
        
    except Exception as e:
        print(f"Error in LSTM model test: {e}")
        import traceback
        traceback.print_exc()