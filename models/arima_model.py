import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    """
    ARIMA model for stock price prediction
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model
        
        Args:
            order: Tuple (p, d, q) for ARIMA parameters
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.is_trained = False
        
    def check_stationarity(self, series, significance_level=0.05):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series data
            significance_level: Significance level for ADF test
            
        Returns:
            bool: True if stationary, False otherwise
        """
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            
            return {
                'is_stationary': p_value < significance_level,
                'p_value': p_value,
                'critical_values': result[4],
                'test_statistic': result[0]
            }
        except Exception as e:
            print(f"Error in stationarity test: {e}")
            return {'is_stationary': False, 'p_value': 1.0}
    
    def find_optimal_order(self, series, max_p=5, max_d=2, max_q=5):
        """
        Find optimal ARIMA order using AIC criterion
        
        Args:
            series: Time series data
            max_p: Maximum p value to test
            max_d: Maximum d value to test
            max_q: Maximum q value to test
            
        Returns:
            tuple: Optimal (p, d, q) order
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Test different combinations
        for p in range(0, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
        
        print(f"Optimal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for ARIMA modeling
        
        Args:
            data: DataFrame with stock data
            target_column: Column to predict
            
        Returns:
            pd.Series: Prepared time series
        """
        # Extract target series
        series = data[target_column].copy()
        
        # Remove any missing values
        series = series.dropna()
        
        # Check for stationarity
        stationarity = self.check_stationarity(series)
        
        if not stationarity['is_stationary']:
            print("Series is not stationary. Consider differencing.")
            print(f"ADF p-value: {stationarity['p_value']:.4f}")
        
        return series
    
    def train(self, data, target_column='Close', auto_order=False):
        """
        Train ARIMA model
        
        Args:
            data: DataFrame with historical stock data
            target_column: Column to predict
            auto_order: Whether to automatically find optimal order
            
        Returns:
            dict: Training results
        """
        try:
            # Prepare data
            series = self.prepare_data(data, target_column)
            
            if len(series) < 50:
                raise ValueError("Insufficient data for ARIMA modeling (minimum 50 observations)")
            
            # Find optimal order if requested
            if auto_order:
                self.order = self.find_optimal_order(series)
            
            # Fit ARIMA model
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_trained = True
            
            # Calculate training metrics
            fitted_values = self.fitted_model.fittedvalues
            residuals = self.fitted_model.resid
            
            # Align series and fitted values
            aligned_series = series[fitted_values.index]
            
            training_results = {
                'success': True,
                'order': self.order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'mse': mean_squared_error(aligned_series, fitted_values),
                'mae': mean_absolute_error(aligned_series, fitted_values),
                'residuals_std': residuals.std(),
                'model_summary': str(self.fitted_model.summary())
            }
            
            print(f"ARIMA{self.order} model trained successfully")
            print(f"AIC: {training_results['aic']:.2f}, BIC: {training_results['bic']:.2f}")
            print(f"Training MSE: {training_results['mse']:.4f}")
            
            return training_results
            
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, steps=7, confidence_interval=True):
        """
        Make predictions using trained ARIMA model
        
        Args:
            steps: Number of steps to forecast
            confidence_interval: Whether to return confidence intervals
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained or self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Make forecast
            forecast_result = self.fitted_model.forecast(
                steps=steps,
                alpha=0.05  # 95% confidence interval
            )
            
            predictions = forecast_result
            
            if confidence_interval:
                # Get confidence intervals
                forecast_ci = self.fitted_model.get_forecast(steps=steps, alpha=0.05)
                confidence_intervals = forecast_ci.conf_int()
                
                result = {
                    'predictions': predictions.tolist(),
                    'lower_ci': confidence_intervals.iloc[:, 0].tolist(),
                    'upper_ci': confidence_intervals.iloc[:, 1].tolist(),
                    'steps': steps
                }
            else:
                result = {
                    'predictions': predictions.tolist(),
                    'steps': steps
                }
            
            return result
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return {'error': str(e)}
    
    def evaluate(self, test_data, target_column='Close'):
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_column: Target column name
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            test_series = test_data[target_column].dropna()
            
            if len(test_series) == 0:
                raise ValueError("No valid test data available")
            
            # Make predictions for test period
            predictions = []
            actuals = []
            
            # Walk-forward validation
            for i in range(len(test_series)):
                if i == 0:
                    # First prediction
                    pred = self.fitted_model.forecast(steps=1)[0]
                else:
                    # Update model with new observation and predict
                    pred = self.fitted_model.forecast(steps=1)[0]
                
                predictions.append(pred)
                actuals.append(test_series.iloc[i])
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            
            # Directional accuracy
            actual_direction = np.diff(actuals) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            evaluation_results = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'predictions': predictions,
                'actuals': actuals
            }
            
            print(f"ARIMA Model Evaluation:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            
            return evaluation_results
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'error': str(e)}
    
    def get_residual_analysis(self):
        """
        Perform residual analysis
        
        Returns:
            dict: Residual analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before residual analysis")
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Jarque-Bera test for normality
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(residuals)
        
        analysis = {
            'residuals': residuals.tolist(),
            'residuals_mean': residuals.mean(),
            'residuals_std': residuals.std(),
            'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
            'jarque_bera_pvalue': jb_pvalue,
            'autocorrelation_significant': lb_test['lb_pvalue'].iloc[-1] < 0.05,
            'normality_rejected': jb_pvalue < 0.05
        }
        
        return analysis
    
    def save_model(self, filepath):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        try:
            self.fitted_model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
        """
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
            self.fitted_model = ARIMAResults.load(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

def compare_arima_orders(data, target_column='Close', orders_to_test=None):
    """
    Compare different ARIMA orders and return the best one
    
    Args:
        data: DataFrame with stock data
        target_column: Column to predict
        orders_to_test: List of (p,d,q) tuples to test
        
    Returns:
        dict: Comparison results
    """
    if orders_to_test is None:
        orders_to_test = [
            (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2),
            (1, 2, 1), (2, 2, 2), (3, 1, 1), (1, 1, 3)
        ]
    
    results = []
    series = data[target_column].dropna()
    
    for order in orders_to_test:
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            results.append({
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'rmse': np.sqrt(mean_squared_error(
                    series[fitted_model.fittedvalues.index],
                    fitted_model.fittedvalues
                ))
            })
            
        except Exception as e:
            print(f"Failed to fit ARIMA{order}: {e}")
            continue
    
    if not results:
        return {'error': 'No models could be fitted'}
    
    # Sort by AIC
    results.sort(key=lambda x: x['aic'])
    
    return {
        'best_order': results[0]['order'],
        'all_results': results
    }