"""
Sentiment Analysis Model for Financial News and Social Media
Uses FinBERT for financial sentiment classification
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import List, Dict, Tuple
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analysis model
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            # Fallback to a lighter model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == "cuda" else -1
            )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit text length for model
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]  # Remove empty texts
        
        if not processed_texts:
            return []
        
        try:
            results = []
            batch_size = 8  # Process in batches to avoid memory issues
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_results = self.sentiment_pipeline(batch)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, list):
                        # FinBERT returns all scores
                        sentiment_scores = {item['label'].lower(): item['score'] for item in result}
                        
                        # Determine dominant sentiment
                        dominant_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
                        confidence = sentiment_scores[dominant_sentiment]
                        
                        results.append({
                            'text': batch[j],
                            'sentiment': dominant_sentiment,
                            'confidence': confidence,
                            'scores': sentiment_scores
                        })
                    else:
                        # Standard sentiment analysis format
                        results.append({
                            'text': batch[j],
                            'sentiment': result['label'].lower(),
                            'confidence': result['score'],
                            'scores': {result['label'].lower(): result['score']}
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return []
    
    def get_numerical_sentiment(self, sentiment: str, confidence: float) -> float:
        """
        Convert sentiment to numerical score
        
        Args:
            sentiment: Sentiment label
            confidence: Confidence score
            
        Returns:
            Numerical sentiment score (-1 to 1)
        """
        sentiment_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'bullish': 1.0,
            'bearish': -1.0
        }
        
        base_score = sentiment_mapping.get(sentiment.lower(), 0.0)
        return base_score * confidence
    
    def aggregate_sentiment(self, sentiment_results: List[Dict], window_hours: int = 24) -> Dict:
        """
        Aggregate sentiment scores over a time window
        
        Args:
            sentiment_results: List of sentiment analysis results
            window_hours: Time window for aggregation
            
        Returns:
            Aggregated sentiment metrics
        """
        if not sentiment_results:
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'confidence': 0.0,
                'total_mentions': 0
            }
        
        # Convert to numerical scores
        numerical_scores = []
        confidences = []
        
        for result in sentiment_results:
            score = self.get_numerical_sentiment(result['sentiment'], result['confidence'])
            numerical_scores.append(score)
            confidences.append(result['confidence'])
        
        avg_sentiment = np.mean(numerical_scores)
        avg_confidence = np.mean(confidences)
        
        # Determine trend
        if avg_sentiment > 0.1:
            trend = 'bullish'
        elif avg_sentiment < -0.1:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'sentiment_trend': trend,
            'confidence': float(avg_confidence),
            'total_mentions': len(sentiment_results),
            'sentiment_distribution': {
                'positive': sum(1 for r in sentiment_results if r['sentiment'] in ['positive', 'bullish']),
                'negative': sum(1 for r in sentiment_results if r['sentiment'] in ['negative', 'bearish']),
                'neutral': sum(1 for r in sentiment_results if r['sentiment'] == 'neutral')
            }
        }

class NewsDataCollector:
    """Collect news data for sentiment analysis"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def get_news_headlines(self, query: str, days_back: int = 1) -> List[Dict]:
        """
        Get news headlines for a given query
        
        Args:
            query: Search query (e.g., company name, ticker)
            days_back: Number of days to look back
            
        Returns:
            List of news articles
        """
        if not self.api_key:
            # Fallback to web scraping if no API key
            return self._scrape_google_news(query, days_back)
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', '')
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _scrape_google_news(self, query: str, days_back: int = 1) -> List[Dict]:
        """
        Fallback method to scrape Google News
        
        Args:
            query: Search query
            days_back: Number of days to look back
            
        Returns:
            List of news articles
        """
        try:
            # Simple Google News scraping (be respectful with rate limiting)
            search_url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US%3Aen"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # This is a simplified scraper - Google News structure changes frequently
            for article in soup.find_all('article')[:10]:  # Limit to 10 articles
                title_elem = article.find('h3')
                if title_elem:
                    articles.append({
                        'title': title_elem.get_text().strip(),
                        'description': '',
                        'content': '',
                        'published_at': datetime.now().isoformat(),
                        'source': 'Google News',
                        'url': ''
                    })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping news: {e}")
            return []

def create_sentiment_features(sentiment_data: List[Dict], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create sentiment features aligned with stock data timestamps
    
    Args:
        sentiment_data: List of sentiment analysis results
        timestamps: Stock data timestamps
        
    Returns:
        DataFrame with sentiment features
    """
    # Create base DataFrame
    df = pd.DataFrame(index=timestamps)
    
    if not sentiment_data:
        # Return neutral sentiment features
        df['sentiment_score'] = 0.0
        df['sentiment_confidence'] = 0.5
        df['news_volume'] = 0
        return df
    
    # Convert sentiment data to DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    
    if 'published_at' in sentiment_df.columns:
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['published_at'])
    else:
        sentiment_df['timestamp'] = pd.Timestamp.now()
    
    # Resample sentiment data to match stock data frequency
    sentiment_df = sentiment_df.set_index('timestamp')
    
    # Create features
    df['sentiment_score'] = 0.0
    df['sentiment_confidence'] = 0.5
    df['news_volume'] = 0
    
    # Aggregate sentiment by time periods
    for timestamp in timestamps:
        # Look for sentiment data within a reasonable window (e.g., 1 hour)
        window_start = timestamp - pd.Timedelta(hours=1)
        window_end = timestamp + pd.Timedelta(hours=1)
        
        mask = (sentiment_df.index >= window_start) & (sentiment_df.index <= window_end)
        window_data = sentiment_df[mask]
        
        if not window_data.empty:
            # Calculate aggregated metrics
            scores = []
            confidences = []
            
            for _, row in window_data.iterrows():
                if 'sentiment_score' in row:
                    scores.append(row['sentiment_score'])
                    confidences.append(row.get('confidence', 0.5))
                elif 'sentiment' in row:
                    # Convert sentiment label to score
                    if row['sentiment'] in ['positive', 'bullish']:
                        scores.append(1.0)
                    elif row['sentiment'] in ['negative', 'bearish']:
                        scores.append(-1.0)
                    else:
                        scores.append(0.0)
                    confidences.append(row.get('confidence', 0.5))
            
            if scores:
                df.loc[timestamp, 'sentiment_score'] = np.mean(scores)
                df.loc[timestamp, 'sentiment_confidence'] = np.mean(confidences)
                df.loc[timestamp, 'news_volume'] = len(scores)
    
    # Forward fill missing values
    df = df.fillna(method='ffill').fillna(0)
    
    return df