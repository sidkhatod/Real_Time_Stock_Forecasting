# utils/sentiment.py - Sentiment analysis utilities

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from config import NEWS_API_CONFIG, SENTIMENT_CONFIG, TWITTER_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for performing sentiment analysis on news and social media data."""
    
    def __init__(self):
        self.news_api_key = NEWS_API_CONFIG['api_key']
        self.finbert_model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self._load_sentiment_model()
    
    def _load_sentiment_model(self):
        """Load FinBERT model for financial sentiment analysis."""
        try:
            model_name = SENTIMENT_CONFIG['model_name']
            logger.info(f"Loading sentiment model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline for easier usage
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            # Fallback to basic VADER sentiment
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("Using VADER sentiment as fallback")
    
    def fetch_news_data(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch news articles for a given ticker.
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of news articles with metadata
        """
        if self.news_api_key == 'your_news_api_key_here':
            logger.warning("News API key not configured. Using sample data.")
            return self._get_sample_news_data(ticker)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Prepare API request
            url = NEWS_API_CONFIG['base_url']
            params = {
                'q': f'{ticker} OR {self._get_company_name(ticker)}',
                'sources': NEWS_API_CONFIG['sources'],
                'language': NEWS_API_CONFIG['language'],
                'sortBy': NEWS_API_CONFIG['sort_by'],
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'pageSize': NEWS_API_CONFIG['page_size'],
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'ticker': ticker
                }
                processed_articles.append(processed_article)
            
            logger.info(f"Fetched {len(processed_articles)} articles for {ticker}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return self._get_sample_news_data(ticker)
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker symbol."""
        ticker_to_company = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook'
        }
        return ticker_to_company.get(ticker, ticker)
    
    def _get_sample_news_data(self, ticker: str) -> List[Dict]:
        """Generate sample news data for testing."""
        sample_articles = [
            {
                'title': f'{ticker} reports strong quarterly earnings',
                'description': f'{ticker} exceeded analyst expectations with strong revenue growth.',
                'content': f'The company showed remarkable performance in Q3 with {ticker} stock rising.',
                'url': 'https://example.com/news1',
                'source': 'Financial Times',
                'published_at': datetime.now().isoformat(),
                'ticker': ticker
            },
            {
                'title': f'{ticker} announces new product launch',
                'description': f'{ticker} unveiled innovative technology that could disrupt the market.',
                'content': f'Industry experts are optimistic about {ticker}\'s latest innovation.',
                'url': 'https://example.com/news2',
                'source': 'Reuters',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'ticker': ticker
            }
        ]
        return sample_articles
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of given texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment scores and labels
        """
        if not texts:
            return []
        
        sentiments = []
        
        try:
            if self.sentiment_pipeline:
                # Use FinBERT for financial sentiment
                for text in texts:
                    if not text or len(text.strip()) == 0:
                        sentiments.append({
                            'label': 'neutral',
                            'score': 0.0,
                            'positive': 0.33,
                            'negative': 0.33,
                            'neutral': 0.34
                        })
                        continue
                    
                    # Truncate text to model's max length
                    max_length = SENTIMENT_CONFIG['max_length']
                    text = text[:max_length] if len(text) > max_length else text
                    
                    result = self.sentiment_pipeline(text)
                    
                    # Parse FinBERT results
                    scores = {item['label'].lower(): item['score'] for item in result[0]}
                    
                    # Determine primary sentiment
                    primary_sentiment = max(scores.keys(), key=lambda k: scores[k])
                    primary_score = scores[primary_sentiment]
                    
                    sentiment_data = {
                        'label': primary_sentiment,
                        'score': primary_score,
                        'positive': scores.get('positive', 0),
                        'negative': scores.get('negative', 0),
                        'neutral': scores.get('neutral', 0)
                    }
                    
                    sentiments.append(sentiment_data)
            
            else:
                # Fallback to VADER
                for text in texts:
                    if not text or len(text.strip()) == 0:
                        sentiments.append({
                            'label': 'neutral',
                            'score': 0.0,
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 0.0
                        })
                        continue
                    
                    scores = self.vader_analyzer.polarity_scores(text)
                    compound = scores['compound']
                    
                    # Determine label based on compound score
                    if compound >= 0.05:
                        label = 'positive'
                    elif compound <= -0.05:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    
                    sentiment_data = {
                        'label': label,
                        'score': abs(compound),
                        'positive': scores['pos'],
                        'negative': scores['neg'],
                        'neutral': scores['neu']
                    }
                    
                    sentiments.append(sentiment_data)
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment for all texts
            sentiments = [{
                'label': 'neutral',
                'score': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34
            } for _ in texts]
        
        return sentiments
    
    def get_ticker_sentiment(self, ticker: str, days_back: int = 7) -> Dict:
        """
        Get aggregated sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        # Fetch news data
        articles = self.fetch_news_data(ticker, days_back)
        
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'sentiment_trend': []
            }
        
        # Prepare texts for analysis
        texts = []
        timestamps = []
        
        for article in articles:
            # Combine title and description for better sentiment analysis
            text = f"{article['title']} {article['description']}"
            texts.append(text)
            timestamps.append(article['published_at'])
        
        # Analyze sentiment
        sentiments = self.analyze_sentiment(texts)
        
        # Aggregate results
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')
        
        # Calculate overall sentiment score
        positive_scores = [s['score'] for s in sentiments if s['label'] == 'positive']
        negative_scores = [s['score'] for s in sentiments if s['label'] == 'negative']
        
        overall_score = 0.0
        if positive_scores or negative_scores:
            pos_avg = np.mean(positive_scores) if positive_scores else 0
            neg_avg = np.mean(negative_scores) if negative_scores else 0
            overall_score = (pos_avg * positive_count - neg_avg * negative_count) / len(sentiments)
        
        # Determine overall sentiment
        if overall_score > 0.1:
            overall_sentiment = 'positive'
        elif overall_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Create sentiment trend (daily aggregation)
        sentiment_trend = self._create_sentiment_trend(sentiments, timestamps)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': overall_score,
            'confidence': np.mean([s['score'] for s in sentiments]),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(articles),
            'sentiment_trend': sentiment_trend,
            'articles_with_sentiment': [
                {**article, 'sentiment': sentiment} 
                for article, sentiment in zip(articles, sentiments)
            ]
        }
    
    def _create_sentiment_trend(self, sentiments: List[Dict], timestamps: List[str]) -> List[Dict]:
        """Create daily sentiment trend data."""
        try:
            # Convert timestamps to dates
            dates = []
            for ts in timestamps:
                try:
                    if isinstance(ts, str):
                        date = pd.to_datetime(ts).date()
                    else:
                        date = ts.date()
                    dates.append(date)
                except:
                    dates.append(datetime.now().date())
            
            # Create DataFrame for easier aggregation
            df = pd.DataFrame({
                'date': dates,
                'sentiment': [s['label'] for s in sentiments],
                'score': [s['score'] for s in sentiments]
            })
            
            # Group by date and calculate daily sentiment
            daily_sentiment = []
            for date, group in df.groupby('date'):
                pos_count = len(group[group['sentiment'] == 'positive'])
                neg_count = len(group[group['sentiment'] == 'negative'])
                neu_count = len(group[group['sentiment'] == 'neutral'])
                
                avg_score = group['score'].mean()
                
                daily_sentiment.append({
                    'date': date.isoformat(),
                    'positive_count': pos_count,
                    'negative_count': neg_count,
                    'neutral_count': neu_count,
                    'avg_score': avg_score,
                    'total_count': len(group)
                })
            
            return sorted(daily_sentiment, key=lambda x: x['date'])
            
        except Exception as e:
            logger.error(f"Error creating sentiment trend: {e}")
            return []

# Utility functions
def get_multi_ticker_sentiment(tickers: List[str], days_back: int = 7) -> Dict[str, Dict]:
    """Get sentiment analysis for multiple tickers."""
    analyzer = SentimentAnalyzer()
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = analyzer.get_ticker_sentiment(ticker, days_back)
        except Exception as e:
            logger.error(f"Error getting sentiment for {ticker}: {e}")
            results[ticker] = {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'total_articles': 0
            }
    
    return results

def calculate_sentiment_signal(sentiment_data: Dict) -> float:
    """
    Calculate trading signal based on sentiment data.
    
    Args:
        sentiment_data: Sentiment analysis results
        
    Returns:
        Signal strength (-1 to 1, where 1 is strong buy, -1 is strong sell)
    """
    try:
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        confidence = sentiment_data.get('confidence', 0)
        total_articles = sentiment_data.get('total_articles', 0)
        
        # Adjust signal based on article volume (more articles = more reliable)
        volume_factor = min(total_articles / 10, 1.0)  # Max factor of 1.0 at 10+ articles
        
        # Calculate final signal
        signal = sentiment_score * confidence * volume_factor
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, signal))
        
    except Exception as e:
        logger.error(f"Error calculating sentiment signal: {e}")
        return 0.0