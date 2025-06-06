import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_loader import fetch_stock_data, get_stock_info
from utils.indicators import calculate_technical_indicators
from models.lstm_model import LSTMPredictor
from utils.sentiment import SentimentAnalyzer
from utils.visualizations import create_candlestick_chart, create_sentiment_chart
import config

# Page config
st.set_page_config(
    page_title="Stock Forecast & Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .bullish { color: #00ff00; }
    .bearish { color: #ff0000; }
    .neutral { color: #ffa500; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None

def main():
    st.markdown('<h1 class="main-header">📈 Real-Time Stock Forecast & Sentiment Dashboard</h1>', unsafe_allow_html=True)
    
    # Auto-refresh every 5 minutes
    st_autorefresh(interval=300000, key="datarefresh")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Stock selection
        selected_stock = st.selectbox(
            "Select Stock",
            config.DEFAULT_STOCKS,
            index=0
        )
        
        # Time period selection
        time_period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            index=4
        )
        
        # Prediction days
        pred_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    try:
        # Fetch stock data
        with st.spinner(f"Loading {selected_stock} data..."):
            stock_data = fetch_stock_data(selected_stock, time_period)
            stock_info = get_stock_info(selected_stock)
        
        if stock_data.empty:
            st.error("No data available for the selected stock.")
            return
        
        # Calculate technical indicators
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Analysis", "🧠 AI Forecast", "💬 Sentiment Analysis", "📊 Trading Signals"])
        
        with tab1:
            show_stock_analysis(stock_data_with_indicators, stock_info, selected_stock)
        
        with tab2:
            show_ai_forecast(stock_data_with_indicators, selected_stock, pred_days)
        
        with tab3:
            show_sentiment_analysis(selected_stock)
        
        with tab4:
            show_trading_signals(stock_data_with_indicators, selected_stock)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please try refreshing the page or selecting a different stock.")

def show_stock_analysis(data, stock_info, ticker):
    """Display stock analysis tab"""
    st.subheader(f"📈 {ticker} Stock Analysis")
    
    # Current price and stats
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
        )
    
    with col2:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    
    with col3:
        high_52w = data['High'].rolling(252).max().iloc[-1]
        st.metric("52W High", f"${high_52w:.2f}")
    
    with col4:
        low_52w = data['Low'].rolling(252).min().iloc[-1]
        st.metric("52W Low", f"${low_52w:.2f}")
    
    # Candlestick chart
    fig = create_candlestick_chart(data, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators summary
    st.subheader("📊 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        current_rsi = data['RSI'].iloc[-1]
        rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
        
        # MACD
        macd_signal = "Bullish" if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else "Bearish"
        st.metric("MACD Signal", macd_signal)
    
    with col2:
        # Bollinger Bands position
        bb_position = ((current_price - data['BB_Lower'].iloc[-1]) / 
                      (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])) * 100
        st.metric("Bollinger Band Position", f"{bb_position:.1f}%")
        
        # Volume trend
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_trend = "High" if data['Volume'].iloc[-1] > avg_volume * 1.5 else "Normal"
        st.metric("Volume Trend", volume_trend)

def show_ai_forecast(data, ticker, pred_days):
    """Display AI forecast tab"""
    st.subheader(f"🧠 LSTM Price Forecast for {ticker}")
    
    try:
        # Initialize LSTM model if not exists
        if st.session_state.lstm_model is None:
            with st.spinner("Initializing LSTM model..."):
                st.session_state.lstm_model = LSTMPredictor()
        
        # Prepare data and make prediction
        with st.spinner("Generating forecast..."):
            forecast = st.session_state.lstm_model.predict(data, pred_days)
        
        if forecast is not None and len(forecast) > 0:
            # Create forecast chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=data.index[-60:],  # Last 60 days
                y=data['Close'].iloc[-60:],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Forecast data
            last_date = data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=pred_days,
                freq='D'
            )
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{ticker} Price Forecast - Next {pred_days} Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            current_price = data['Close'].iloc[-1]
            forecast_price = forecast[-1]
            price_change = forecast_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric(
                    f"Predicted Price ({pred_days}d)",
                    f"${forecast_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
                )
            
            with col3:
                direction = "📈 Bullish" if price_change > 0 else "📉 Bearish"
                st.metric("Trend Direction", direction)
            
            # Model confidence and details
            st.subheader("📋 Model Details")
            st.info(f"""
            **Model Type**: LSTM (Long Short-Term Memory)
            **Training Features**: OHLCV + Technical Indicators
            **Prediction Horizon**: {pred_days} days
            **Note**: Predictions are based on historical patterns and should not be used as sole investment advice.
            """)
            
        else:
            st.warning("Unable to generate forecast. Please try with different parameters.")
            
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

def show_sentiment_analysis(ticker):
    """Display sentiment analysis tab"""
    st.subheader(f"💬 Sentiment Analysis for {ticker}")
    
    try:
        with st.spinner("Analyzing market sentiment..."):
            # Get sentiment data
            sentiment_data = st.session_state.sentiment_analyzer.get_stock_sentiment(ticker)
        
        if sentiment_data:
            # Overall sentiment score
            overall_sentiment = sentiment_data.get('overall_sentiment', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Sentiment", sentiment_label)
            
            with col2:
                st.metric("Sentiment Score", f"{overall_sentiment:.3f}")
            
            with col3:
                confidence = sentiment_data.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Sentiment breakdown
            if 'sentiment_breakdown' in sentiment_data:
                st.subheader("📊 Sentiment Breakdown")
                breakdown = sentiment_data['sentiment_breakdown']
                
                fig = px.pie(
                    values=list(breakdown.values()),
                    names=list(breakdown.keys()),
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent news/tweets
            if 'recent_news' in sentiment_data:
                st.subheader("📰 Recent News Analysis")
                news_items = sentiment_data['recent_news'][:5]  # Show top 5
                
                for item in news_items:
                    with st.expander(f"📰 {item.get('title', 'News Item')}"):
                        st.write(f"**Sentiment**: {item.get('sentiment', 'N/A')}")
                        st.write(f"**Score**: {item.get('score', 'N/A'):.3f}")
                        st.write(f"**Summary**: {item.get('summary', 'No summary available')}")
            
            # Sentiment timeline
            if 'sentiment_timeline' in sentiment_data:
                st.subheader("📈 Sentiment Timeline")
                timeline = sentiment_data['sentiment_timeline']
                
                fig = create_sentiment_chart(timeline)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No sentiment data available. This could be due to API limitations or lack of recent news.")
            
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")

def show_trading_signals(data, ticker):
    """Display trading signals tab"""
    st.subheader(f"📊 Trading Signals for {ticker}")
    
    # Generate signals
    signals = generate_trading_signals(data)
    
    # Current signal
    current_signal = signals['signal'].iloc[-1]
    signal_strength = signals['strength'].iloc[-1]
    
    # Display current recommendation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "green" if current_signal == "BUY" else "red" if current_signal == "SELL" else "orange"
        st.markdown(f'<h2 style="color:{color};">🎯 {current_signal}</h2>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Signal Strength", f"{signal_strength:.1%}")
    
    with col3:
        risk_level = "Low" if signal_strength > 0.7 else "Medium" if signal_strength > 0.4 else "High"
        st.metric("Risk Level", risk_level)
    
    # Signal explanation
    st.subheader("📋 Signal Analysis")
    
    # Technical analysis summary
    rsi = data['RSI'].iloc[-1]
    macd_signal = data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]
    bb_position = ((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / 
                   (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]))
    
    factors = []
    
    if rsi < 30:
        factors.append("✅ RSI indicates oversold condition (potential buy)")
    elif rsi > 70:
        factors.append("⚠️ RSI indicates overbought condition (potential sell)")
    else:
        factors.append("➖ RSI in neutral range")
    
    if macd_signal:
        factors.append("✅ MACD above signal line (bullish)")
    else:
        factors.append("⚠️ MACD below signal line (bearish)")
    
    if bb_position < 0.2:
        factors.append("✅ Price near lower Bollinger Band (potential buy)")
    elif bb_position > 0.8:
        factors.append("⚠️ Price near upper Bollinger Band (potential sell)")
    else:
        factors.append("➖ Price within Bollinger Bands")
    
    for factor in factors:
        st.write(factor)
    
    # Signal history chart
    st.subheader("📈 Signal History")
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index[-60:],
        y=data['Close'].iloc[-60:],
        mode='lines',
        name='Price',
        yaxis='y'
    ))
    
    # Buy/Sell signals
    buy_signals = signals[signals['signal'] == 'BUY'].tail(10)
    sell_signals = signals[signals['signal'] == 'SELL'].tail(10)
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=data.loc[buy_signals.index, 'Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            yaxis='y'
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=data.loc[sell_signals.index, 'Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            yaxis='y'
        ))
    
    fig.update_layout(
        title=f"{ticker} Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.warning("⚠️ **Disclaimer**: Trading signals are for educational purposes only. Always do your own research and consider consulting with a financial advisor before making investment decisions.")

def generate_trading_signals(data):
    """Generate trading signals based on technical indicators"""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 'HOLD'
    signals['strength'] = 0.5
    
    # RSI signals
    rsi_buy = data['RSI'] < 30
    rsi_sell = data['RSI'] > 70
    
    # MACD signals
    macd_buy = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
    macd_sell = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
    
    # Bollinger Bands signals
    bb_buy = data['Close'] < data['BB_Lower']
    bb_sell = data['Close'] > data['BB_Upper']
    
    # Combine signals
    buy_score = rsi_buy.astype(int) + macd_buy.astype(int) + bb_buy.astype(int)
    sell_score = rsi_sell.astype(int) + macd_sell.astype(int) + bb_sell.astype(int)
    
    # Generate final signals
    signals.loc[buy_score >= 2, 'signal'] = 'BUY'
    signals.loc[sell_score >= 2, 'signal'] = 'SELL'
    
    # Calculate strength
    signals['strength'] = np.maximum(buy_score, sell_score) / 3.0
    signals['strength'] = signals['strength'].fillna(0.5)
    
    return signals

if __name__ == "__main__":
    main()