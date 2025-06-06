# utils/data_loader.py - Stock Data Ingestion Module

import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import time
from functools import wraps

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    Comprehensive stock data loader with caching, real-time updates,
    and robust error handling.
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.last_update = {}
        
    def _cache_key(self, ticker: str, period: str, interval: str) -> str:
        """Generate cache key for data storage."""
        return f"{ticker}_{period}_{interval}"
    
    def _is_cache_valid(self, cache_key: str, cache_duration: int = 300) -> bool:
        """Check if cached data is still valid."""
        if not self.cache_enabled or cache_key not in self.last_update:
            return False
        
        time_elapsed = time.time() - self.last_update[cache_key]
        return time_elapsed < cache_duration
    
    def _retry_on_failure(max_retries: int = 3, delay: float = 1.0):
        """Decorator for retrying failed API calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                            raise
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying...")
                        time.sleep(delay * (attempt + 1))
                return None
            return wrapper
        return decorator
    
    @_retry_on_failure(max_retries=3, delay=1.0)
    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str = '1y', 
        interval: str = '1d',
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given ticker with caching support.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Force refresh cache
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        cache_key = self._cache_key(ticker, period, interval)
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_key):
            logger.info(f"Returning cached data for {ticker}")
            return self.cache[cache_key].copy()
        
        try:
            logger.info(f"Fetching fresh data for {ticker} (period={period}, interval={interval})")
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Add additional metrics
            data = self._add_basic_metrics(data)
            
            # Add ticker column for multi-stock analysis
            data['Ticker'] = ticker
            
            # Cache the data
            if self.cache_enabled:
                self.cache[cache_key] = data.copy()
                self.last_update[cache_key] = time.time()
            
            # Save to file for persistence
            self.save_data_to_file(data, ticker)
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            
            # Try to load from file as fallback
            file_path = get_data_file_path(ticker)
            if os.path.exists(file_path):
                logger.info(f"Loading fallback data from file for {ticker}")
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            raise e
    
    def _add_basic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived metrics to the stock data."""
        if data.empty:
            return data
        
        data = data.copy()
        
        # Price changes
        data['Price_Change'] = data['Close'].diff()
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100
        
        # Price ranges
        data['Daily_Range'] = data['High'] - data['Low']
        data['Daily_Range_Pct'] = (data['Daily_Range'] / data['Close']) * 100
        
        # Gap analysis
        data['Gap'] = data['Open'] - data['Close'].shift(1)
        data['Gap_Pct'] = (data['Gap'] / data['Close'].shift(1)) * 100
        
        # Volatility (rolling 20-day)
        data['Volatility_20d'] = data['Price_Change_Pct'].rolling(window=20).std()
        
        # Volume metrics
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # High/Low analysis
        data['High_20d'] = data['High'].rolling(window=20).max()
        data['Low_20d'] = data['Low'].rolling(window=20).min()
        data['Distance_from_High'] = ((data['Close'] - data['High_20d']) / data['High_20d']) * 100
        data['Distance_from_Low'] = ((data['Close'] - data['Low_20d']) / data['Low_20d']) * 100
        
        return data
    
    def fetch_multiple_stocks(
        self, 
        tickers: List[str], 
        period: str = '1y', 
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        results = {}
        
        for ticker in tickers:
            try:
                data = self.fetch_stock_data(ticker, period, interval)
                results[ticker] = data
                logger.info(f"Successfully loaded {ticker}")
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
                continue
        
        return results
    
    def get_real_time_data(self, ticker: str) -> Dict:
        """
        Get real-time stock data and key metrics.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with real-time data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get latest price data
            hist = stock.history(period='1d', interval='1m')
            if hist.empty:
                hist = stock.history(period='5d', interval='1d')
            
            latest = hist.iloc[-1] if not hist.empty else None
            
            real_time_data = {
                'symbol': ticker,
                'current_price': latest['Close'] if latest is not None else info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open_price': latest['Open'] if latest is not None else info.get('open', 0),
                'day_high': latest['High'] if latest is not None else info.get('dayHigh', 0),
                'day_low': latest['Low'] if latest is not None else info.get('dayLow', 0),
                'volume': latest['Volume'] if latest is not None else info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'timestamp': datetime.now()
            }
            
            # Calculate derived metrics
            if real_time_data['current_price'] and real_time_data['previous_close']:
                price_change = real_time_data['current_price'] - real_time_data['previous_close']
                real_time_data['price_change'] = price_change
                real_time_data['price_change_pct'] = (price_change / real_time_data['previous_close']) * 100
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {ticker}: {e}")
            return {}
    
    def save_data_to_file(self, data: pd.DataFrame, ticker: str) -> None:
        """Save data to CSV file for persistence."""
        try:
            file_path = get_data_file_path(ticker)
            data.to_csv(file_path)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to file: {e}")
    
    def load_data_from_file(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        try:
            file_path = get_data_file_path(ticker)
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Data loaded from {file_path}")
                return data
            return None
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
            return None
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get comprehensive stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Clean and organize the info
            stock_info = {
                'basic_info': {
                    'symbol': info.get('symbol', ticker),
                    'company_name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'country': info.get('country', ''),
                    'website': info.get('website', ''),
                    'description': info.get('longBusinessSummary', ''),
                },
                'valuation': {
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                },
                'financial_metrics': {
                    'revenue': info.get('totalRevenue', 0),
                    'gross_profit': info.get('grossProfits', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'return_on_equity': info.get('returnOnEquity', 0),
                    'return_on_assets': info.get('returnOnAssets', 0),
                },
                'dividend_info': {
                    'dividend_yield': info.get('dividendYield', 0),
                    'dividend_rate': info.get('dividendRate', 0),
                    'payout_ratio': info.get('payoutRatio', 0),
                    'ex_dividend_date': info.get('exDividendDate', ''),
                }
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            return {}
    
    def clear_cache(self, ticker: str = None) -> None:
        """Clear cache for specific ticker or all tickers."""
        if ticker:
            # Clear cache for specific ticker
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(ticker)]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.last_update:
                    del self.last_update[key]
            logger.info(f"Cache cleared for {ticker}")
        else:
            # Clear all cache
            self.cache.clear()
            self.last_update.clear()
            logger.info("All cache cleared")


# Utility functions for easy access
def get_stock_data(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """Quick function to get stock data."""
    loader = StockDataLoader()
    return loader.fetch_stock_data(ticker, period, interval)

def get_multiple_stocks_data(tickers: List[str], period: str = '1y', interval: str = '1d') -> Dict[str, pd.DataFrame]:
    """Quick function to get multiple stocks data."""
    loader = StockDataLoader()
    return loader.fetch_multiple_stocks(tickers, period, interval)

def get_real_time_quote(ticker: str) -> Dict:
    """Quick function to get real-time quote."""
    loader = StockDataLoader()
    return loader.get_real_time_data(ticker)


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader()
    
    # Test single stock
    print("Testing single stock data fetch...")
    try:
        aapl_data = loader.fetch_stock_data('AAPL', period='1mo', interval='1d')
        print(f"AAPL data shape: {aapl_data.shape}")
        print(f"AAPL latest close: ${aapl_data['Close'].iloc[-1]:.2f}")
        print(f"AAPL columns: {list(aapl_data.columns)}")
    except Exception as e:
        print(f"Error testing single stock: {e}")
    
    # Test real-time data
    print("\nTesting real-time data...")
    try:
        real_time = loader.get_real_time_data('AAPL')
        print(f"Real-time AAPL: ${real_time.get('current_price', 0):.2f}")
        print(f"Change: {real_time.get('price_change_pct', 0):.2f}%")
    except Exception as e:
        print(f"Error testing real-time data: {e}")
    
    # Test multiple stocks
    print("\nTesting multiple stocks...")
    try:
        multiple_data = loader.fetch_multiple_stocks(['AAPL', 'GOOGL'], period='5d')
        print(f"Loaded {len(multiple_data)} stocks")
        for ticker, data in multiple_data.items():
            print(f"{ticker}: {data.shape[0]} rows")
    except Exception as e:
        print(f"Error testing multiple stocks: {e}")