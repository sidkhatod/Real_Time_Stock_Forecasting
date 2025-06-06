import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_candlestick_chart(data, ticker, show_volume=True, show_indicators=True):
    """
    Create an interactive candlestick chart with technical indicators
    
    Args:
        data: DataFrame with OHLCV data and technical indicators
        ticker: Stock ticker symbol
        show_volume: Whether to show volume subplot
        show_indicators: Whether to show technical indicators
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Determine subplot structure
    rows = 2 if show_volume else 1
    subplot_titles = [f"{ticker} Stock Price"]
    if show_volume:
        subplot_titles.append("Volume")
    
    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
        row_heights=[0.7, 0.3] if show_volume else [1.0]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f"{ticker} OHLC",
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # Add technical indicators if available and requested
    if show_indicators:
        # Moving averages
        if 'EMA_12' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='orange', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        if 'EMA_26' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='purple', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.6
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    opacity=0.6
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='blue', width=1, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # Volume chart
    if show_volume and 'Volume' in data.columns:
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Remove rangeslider for cleaner look
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def create_rsi_chart(data, ticker):
    """
    Create RSI chart with overbought/oversold levels
    
    Args:
        data: DataFrame with RSI data
        ticker: Stock ticker symbol
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        )
    )
    
    # Overbought line (70)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Overbought (70)"
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Oversold (30)"
    )
    
    # Neutral line (50)
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="Neutral (50)"
    )
    
    fig.update_layout(
        title=f"{ticker} RSI (Relative Strength Index)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig

def create_macd_chart(data, ticker):
    """
    Create MACD chart with signal line and histogram
    
    Args:
        data: DataFrame with MACD data
        ticker: Stock ticker symbol
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"{ticker} MACD", "MACD Histogram"],
        row_heights=[0.7, 0.3]
    )
    
    # MACD line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Signal line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # MACD Histogram
    if 'MACD_Histogram' in data.columns:
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title=f"{ticker} MACD Analysis",
        height=400,
        showlegend=True
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """
    Create sentiment timeline chart
    
    Args:
        sentiment_data: DataFrame or dict with sentiment data over time
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    if isinstance(sentiment_data, dict):
        # Convert dict to DataFrame
        df = pd.DataFrame(list(sentiment_data.items()), columns=['date', 'sentiment'])
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = sentiment_data.copy()
    
    # Color mapping for sentiment
    colors = []
    for sentiment in df['sentiment']:
        if sentiment > 0.1:
            colors.append('green')
        elif sentiment < -0.1:
            colors.append('red')
        else:
            colors.append('orange')
    
    # Sentiment line chart
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2),
            marker=dict(color=colors, size=6)
        )
    )
    
    # Neutral line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Neutral"
    )
    
    # Positive/Negative zones
    fig.add_hrect(
        y0=0.1, y1=1,
        fillcolor="green", opacity=0.1,
        annotation_text="Positive", annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=-1, y1=-0.1,
        fillcolor="red", opacity=0.1,
        annotation_text="Negative", annotation_position="bottom left"
    )
    
    fig.update_layout(
        title="Sentiment Analysis Timeline",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1]),
        height=400
    )
    
    return fig

def create_correlation_heatmap(data, features):
    """
    Create correlation heatmap for selected features
    
    Args:
        data: DataFrame with features
        features: List of feature names to include
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        )
    )
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=600,
        height=600
    )
    
    return fig

def create_volume_profile(data, bins=50):
    """
    Create volume profile chart
    
    Args:
        data: DataFrame with OHLCV data
        bins: Number of price bins
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Calculate price range
    price_min = data['Low'].min()
    price_max = data['High'].max()
    price_bins = np.linspace(price_min, price_max, bins)
    
    # Calculate volume at each price level
    volume_profile = []
    for i in range(len(price_bins) - 1):
        low_bin = price_bins[i]
        high_bin = price_bins[i + 1]
        
        # Find bars that overlap with this price range
        mask = (data['Low'] <= high_bin) & (data['High'] >= low_bin)
        volume_at_level = data[mask]['Volume'].sum()
        
        volume_profile.append({
            'price': (low_bin + high_bin) / 2,
            'volume': volume_at_level
        })
    
    volume_df = pd.DataFrame(volume_profile)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=volume_df['volume'],
            y=volume_df['price'],
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(58, 71, 80, 0.6)'
        )
    )
    
    fig.update_layout(
        title="Volume Profile",
        xaxis_title="Volume",
        yaxis_title="Price ($)",
        height=600
    )
    
    return fig

def create_performance_comparison(returns_data, benchmarks=None):
    """
    Create performance comparison chart
    
    Args:
        returns_data: Dict with asset names as keys and return series as values
        benchmarks: List of benchmark names to highlight
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (name, returns) in enumerate(returns_data.items()):
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        line_width = 3 if benchmarks and name in benchmarks else 2
        
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cumulative_returns,
                mode='lines',
                name=name,
                line=dict(
                    color=colors[i % len(colors)],
                    width=line_width
                )
            )
        )
    
    fig.update_layout(
        title="Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        height=500,
        hovermode='x unified'
    )
    
    return fig