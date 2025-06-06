# config.py - Configuration file for the Stock Dashboard project

import os
from datetime import datetime, timedelta

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Project directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, UTILS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# STOCK DATA CONFIGURATION
# =============================================================================

# Default stock symbols to track
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']

# Data fetching parameters
STOCK_DATA_CONFIG = {
    'default_period': '1y',          # 1 year of historical data
    'default_interval': '1d',        # Daily intervals
    'real_time_interval': '1m',      # 1-minute intervals for real-time
    'cache_duration': 300,           # Cache data for 5 minutes
}

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'sma_periods': [20, 50, 200],    # Simple Moving Averages
    'ema_periods': [12, 26],         # Exponential Moving Averages
    'rsi_period': 14,                # RSI period
    'macd_fast': 12,                 # MACD fast period
    'macd_slow': 26,                 # MACD slow period
    'macd_signal': 9,                # MACD signal period
    'bb_period': 20,                 # Bollinger Bands period
    'bb_std': 2,                     # Bollinger Bands standard deviation
}

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

# LSTM Model parameters
LSTM_CONFIG = {
    'sequence_length': 60,           # Look back 60 days
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'validation_split': 0.2,
    'test_split': 0.2,
}

# Feature scaling
SCALING_CONFIG = {
    'method': 'minmax',              # 'minmax' or 'standard'
    'feature_range': (0, 1),         # For MinMax scaling
}

# =============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================

# News API configuration (you'll need to get your own API key)
NEWS_API_CONFIG = {
    'api_key': os.getenv('NEWS_API_KEY', 'your_news_api_key_here'),
    'base_url': 'https://newsapi.org/v2/everything',
    'sources': 'bloomberg,reuters,financial-times,cnbc,marketwatch',
    'language': 'en',
    'sort_by': 'publishedAt',
    'page_size': 100,
}

# Twitter API configuration (optional - requires Twitter API v2 access)
TWITTER_CONFIG = {
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', 'your_twitter_bearer_token'),
    'max_tweets': 100,
    'tweet_fields': 'created_at,public_metrics,context_annotations',
}

# Sentiment model configuration
SENTIMENT_CONFIG = {
    'model_name': 'ProsusAI/finbert',  # FinBERT for financial sentiment
    'batch_size': 16,
    'max_length': 512,
    'sentiment_threshold': 0.1,       # Confidence threshold
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

# Streamlit configuration
DASHBOARD_CONFIG = {
    'page_title': 'Real-Time Stock Forecast & Sentiment Dashboard',
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'refresh_interval': 30,           # Auto-refresh every 30 seconds
}

# Chart configuration
CHART_CONFIG = {
    'candlestick_colors': {
        'increasing': '#00ff00',
        'decreasing': '#ff0000',
    },
    'forecast_color': '#ff6b6b',
    'sentiment_colors': {
        'positive': '#4CAF50',
        'neutral': '#FFC107',
        'negative': '#F44336',
    },
}

# =============================================================================
# TRADING SIGNALS CONFIGURATION
# =============================================================================

# Signal generation parameters
SIGNAL_CONFIG = {
    'price_weight': 0.6,             # Weight for price-based signals
    'sentiment_weight': 0.4,         # Weight for sentiment-based signals
    'signal_threshold': {
        'strong_buy': 0.8,
        'buy': 0.6,
        'hold': 0.4,
        'sell': 0.2,
        'strong_sell': 0.0,
    },
}

# Risk management
RISK_CONFIG = {
    'max_position_size': 0.1,        # Maximum 10% of portfolio per position
    'stop_loss_pct': 0.05,           # 5% stop loss
    'take_profit_pct': 0.15,         # 15% take profit
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(PROJECT_ROOT, 'logs', 'dashboard.log'),
}

# Create logs directory
os.makedirs(os.path.dirname(LOGGING_CONFIG['log_file']), exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_file_path(ticker: str, file_type: str = 'csv') -> str:
    """Get the file path for storing ticker data."""
    return os.path.join(DATA_DIR, f"{ticker}.{file_type}")

def get_model_file_path(ticker: str, model_type: str = 'lstm') -> str:
    """Get the file path for storing model files."""
    return os.path.join(MODELS_DIR, f"{ticker}_{model_type}_model.h5")

def get_date_range(days_back: int = 365) -> tuple:
    """Get start and end dates for data fetching."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['NEWS_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Some features may not work properly.")
    
    return len(missing_vars) == 0

if __name__ == "__main__":
    # Test configuration
    print("Stock Dashboard Configuration")
    print("=" * 40)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Default Tickers: {DEFAULT_TICKERS}")
    print(f"Environment Valid: {validate_environment()}")