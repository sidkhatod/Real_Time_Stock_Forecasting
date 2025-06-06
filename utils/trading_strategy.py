"""
Trading Strategy Module
Combines technical analysis, ML predictions, and sentiment analysis to generate trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Signal(Enum):
    """Trading signal enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    signal: Signal
    confidence: float
    price: float
    timestamp: pd.Timestamp
    reasoning: List[str]
    technical_score: float
    ml_score: float
    sentiment_score: float
    risk_level: str

class TradingStrategy:
    """
    Main trading strategy class that combines multiple signals
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize trading strategy
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default strategy configuration"""
        return {
            # Signal weights
            'technical_weight': 0.4,
            'ml_weight': 0.4,
            'sentiment_weight': 0.2,
            
            # Thresholds
            'buy_threshold': 0.6,
            'sell_threshold': -0.6,
            'confidence_threshold': 0.5,
            
            # Risk management
            'max_position_size': 0.1,  # 10% of portfolio
            'stop_loss_pct': 0.05,     # 5% stop loss
            'take_profit_pct': 0.15,   # 15% take profit
            
            # Technical indicator parameters
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_threshold': 0.1,
            
            # ML model parameters
            'ml_confidence_threshold': 0.7,
            'prediction_horizon_days': 5,
            
            # Sentiment parameters
            'sentiment_threshold': 0.3,
            'news_volume_threshold': 5
        }
    
    def generate_signal(self, 
                       current_data: pd.Series,
                       technical_indicators: Dict,
                       ml_prediction: Dict,
                       sentiment_data: Dict,
                       historical_data: pd.DataFrame = None) -> TradingSignal:
        """
        Generate trading signal based on multiple factors
        
        Args:
            current_data: Current price data
            technical_indicators: Technical analysis indicators
            ml_prediction: ML model prediction
            sentiment_data: Sentiment analysis results
            historical_data: Historical price data for context
            
        Returns:
            TradingSignal object
        """
        
        # Calculate individual scores
        technical_score = self._calculate_technical_score(technical_indicators)
        ml_score = self._calculate_ml_score(ml_prediction)
        sentiment_score = self._calculate_sentiment_score(sentiment_data)
        
        # Combine scores with weights
        combined_score = (
            technical_score * self.config['technical_weight'] +
            ml_score * self.config['ml_weight'] +
            sentiment_score * self.config['sentiment_weight']
        )
        
        # Generate signal based on combined score
        if combined_score >= self.config['buy_threshold']:
            signal = Signal.BUY
        elif combined_score <= self.config['sell_threshold']:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        # Calculate confidence
        confidence = min(abs(combined_score), 1.0)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            signal, technical_indicators, ml_prediction, sentiment_data, 
            technical_score, ml_score, sentiment_score
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(
            technical_indicators, ml_prediction, sentiment_data, historical_data
        )
        
        return TradingSignal(
            signal=signal,
            confidence=confidence,
            price=current_data.get('close', 0),
            timestamp=pd.Timestamp.now(),
            reasoning=reasoning,
            technical_score=technical_score,
            ml_score=ml_score,
            sentiment_score=sentiment_score,
            risk_level=risk_level
        )
    
    def _calculate_technical_score(self, indicators: Dict) -> float:
        """
        Calculate technical analysis score
        
        Args:
            indicators: Technical indicators dictionary
            
        Returns:
            Technical score (-1 to 1)
        """
        score = 0.0
        count = 0
        
        # RSI analysis
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < self.config['rsi_oversold']:
                score += 0.5  # Oversold -> potential buy
            elif rsi > self.config['rsi_overbought']:
                score -= 0.5  # Overbought -> potential sell
            count += 1
        
        # MACD analysis
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_diff = indicators['macd'] - indicators['macd_signal']
            if macd_diff > self.config['macd_signal_threshold']:
                score += 0.3  # Bullish crossover
            elif macd_diff < -self.config['macd_signal_threshold']:
                score -= 0.3  # Bearish crossover
            count += 1
        
        # Moving Average analysis
        if 'sma_20' in indicators and 'sma_50' in indicators:
            if indicators['sma_20'] > indicators['sma_50']:
                score += 0.2  # Short MA above long MA -> bullish
            else:
                score -= 0.2  # Short MA below long MA -> bearish
            count += 1
        
        # Bollinger Bands analysis
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'close']):
            close_price = indicators['close']
            bb_position = (close_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            if bb_position < 0.2:
                score += 0.3  # Near lower band -> potential buy
            elif bb_position > 0.8:
                score -= 0.3  # Near upper band -> potential sell
            count += 1
        
        # Volume analysis
        if 'volume_sma' in indicators and 'volume' in indicators:
            volume_ratio = indicators['volume'] / indicators['volume_sma']
            if volume_ratio > 1.5:  # High volume
                # High volume strengthens the signal
                score *= 1.2
            count += 1
        
        return score / max(count, 1)  # Normalize by number of indicators
    
    def _calculate_ml_score(self, ml_prediction: Dict) -> float:
        """
        Calculate ML model score
        
        Args:
            ml_prediction: ML prediction dictionary
            
        Returns:
            ML score (-1 to 1)
        """
        if not ml_prediction or 'prediction' not in ml_prediction:
            return 0.0
        
        prediction = ml_prediction['prediction']
        confidence = ml_prediction.get('confidence', 0.5)
        current_price = ml_prediction.get('current_price', 0)
        
        if current_price == 0:
            return 0.0
        
        # Calculate price change percentage
        price_change_pct = (prediction - current_price) / current_price
        
        # Scale the score based on price change and confidence
        score = np.tanh(price_change_pct * 10) * confidence
        
        # Only consider predictions with sufficient confidence
        if confidence < self.config['ml_confidence_threshold']:
            score *= 0.5
        
        return float(np.clip(score, -1, 1))
    
    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """
        Calculate sentiment score
        
        Args:
            sentiment_data: Sentiment analysis results
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if not sentiment_data:
            return 0.0
        
        avg_sentiment = sentiment_data.get('avg_sentiment', 0.0)
        confidence = sentiment_data.get('confidence', 0.5)
        news_volume = sentiment_data.get('total_mentions', 0)
        
        # Base sentiment score
        score = avg_sentiment * confidence
        
        # Boost score if there's significant news volume
        if news_volume >= self.config['news_volume_threshold']:
            score *= 1.2
        elif news_volume == 0:
            score *= 0.5  # Reduce impact if no news
        
        return float(np.clip(score, -1, 1))
    
    def _generate_reasoning(self, 
                          signal: Signal,
                          technical_indicators: Dict,
                          ml_prediction: Dict,
                          sentiment_data: Dict,
                          technical_score: float,
                          ml_score: float,
                          sentiment_score: float) -> List[str]:
        """
        Generate human-readable reasoning for the signal
        
        Args:
            signal: Generated trading signal
            technical_indicators: Technical indicators
            ml_prediction: ML prediction data
            sentiment_data: Sentiment data
            technical_score: Technical analysis score
            ml_score: ML model score
            sentiment_score: Sentiment score
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Technical analysis reasoning
        if abs(technical_score) > 0.3:
            if technical_score > 0:
                reasoning.append(f"Technical indicators are bullish (score: {technical_score:.2f})")
                
                if 'rsi' in technical_indicators and technical_indicators['rsi'] < 30:
                    reasoning.append("RSI indicates oversold conditions")
                if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
                    if technical_indicators['macd'] > technical_indicators['macd_signal']:
                        reasoning.append("MACD shows bullish momentum")
            else:
                reasoning.append(f"Technical indicators are bearish (score: {technical_score:.2f})")
                
                if 'rsi' in technical_indicators and technical_indicators['rsi'] > 70:
                    reasoning.append("RSI indicates overbought conditions")
                if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
                    if technical_indicators['macd'] < technical_indicators['macd_signal']:
                        reasoning.append("MACD shows bearish momentum")
        
        # ML prediction reasoning
        if abs(ml_score) > 0.3:
            confidence = ml_prediction.get('confidence', 0) * 100
            if ml_score > 0:
                reasoning.append(f"AI model predicts price increase (confidence: {confidence:.1f}%)")
            else:
                reasoning.append(f"AI model predicts price decrease (confidence: {confidence:.1f}%)")
        
        # Sentiment reasoning
        if abs(sentiment_score) > 0.2:
            news_volume = sentiment_data.get('total_mentions', 0)
            if sentiment_score > 0:
                reasoning.append(f"Market sentiment is positive with {news_volume} news mentions")
            else:
                reasoning.append(f"Market sentiment is negative with {news_volume} news mentions")
        
        # Overall signal reasoning
        if signal == Signal.BUY:
            reasoning.append("🟢 Recommendation: BUY - Multiple indicators align for potential upward movement")
        elif signal == Signal.SELL:
            reasoning.append("🔴 Recommendation: SELL - Multiple indicators suggest potential downward movement")
        else:
            reasoning.append("🟡 Recommendation: HOLD - Mixed signals suggest waiting for clearer direction")
        
        return reasoning
    
    def _assess_risk_level(self, 
                          technical_indicators: Dict,
                          ml_prediction: Dict,
                          sentiment_data: Dict,
                          historical_data: pd.DataFrame = None) -> str:
        """
        Assess risk level of the trading signal
        
        Args:
            technical_indicators: Technical indicators
            ml_prediction: ML prediction data
            sentiment_data: Sentiment data
            historical_data: Historical price data
            
        Returns:
            Risk level string
        """
        risk_factors = 0
        
        # Volatility risk
        if historical_data is not None and len(historical_data) > 20:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            if volatility > 0.4:  # High volatility
                risk_factors += 2
            elif volatility > 0.25:  # Medium volatility
                risk_factors += 1
        
        # Technical indicator conflicts
        conflicting_signals = 0
        if 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
            if 30 < rsi < 70:  # Neutral RSI
                conflicting_signals += 1
        
        if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
            macd_diff = abs(technical_indicators['macd'] - technical_indicators['macd_signal'])
            if macd_diff < 0.05:  # Weak MACD signal
                conflicting_signals += 1
        
        if conflicting_signals >= 2:
            risk_factors += 1
        
        # ML prediction uncertainty
        ml_confidence = ml_prediction.get('confidence', 0) if ml_prediction else 0
        if ml_confidence < 0.6:
            risk_factors += 1
        
        # Sentiment uncertainty
        sentiment_confidence = sentiment_data.get('confidence', 0) if sentiment_data else 0
        news_volume = sentiment_data.get('total_mentions', 0) if sentiment_data else 0
        
        if sentiment_confidence < 0.5 or news_volume < 3:
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 4:
            return "HIGH"
        elif risk_factors >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_position_size(self, 
                              signal: TradingSignal,
                              portfolio_value: float,
                              current_price: float) -> Dict:
        """
        Calculate recommended position size based on risk management
        
        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_price: Current stock price
            
        Returns:
            Position sizing recommendations
        """
        if signal.signal == Signal.HOLD:
            return {
                'position_size': 0,
                'shares': 0,
                'dollar_amount': 0,
                'risk_pct': 0
            }
        
        # Base position size
        base_position_pct = self.config['max_position_size']
        
        # Adjust based on confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on risk level
        risk_multipliers = {
            'LOW': 1.0,
            'MEDIUM': 0.7,
            'HIGH': 0.4
        }
        risk_multiplier = risk_multipliers.get(signal.risk_level, 0.5)
        
        # Calculate final position size
        final_position_pct = base_position_pct * confidence_multiplier * risk_multiplier
        dollar_amount = portfolio_value * final_position_pct
        shares = int(dollar_amount / current_price) if current_price > 0 else 0
        
        return {
            'position_size_pct': final_position_pct * 100,
            'shares': shares,
            'dollar_amount': dollar_amount,
            'risk_pct': final_position_pct * 100,
            'stop_loss_price': current_price * (1 - self.config['stop_loss_pct']) if signal.signal == Signal.BUY else current_price * (1 + self.config['stop_loss_pct']),
            'take_profit_price': current_price * (1 + self.config['take_profit_pct']) if signal.signal == Signal.BUY else current_price * (1 - self.config['take_profit_pct'])
        }
    
    def backtest_strategy(self, 
                         historical_data: pd.DataFrame,
                         technical_indicators: pd.DataFrame,
                         ml_predictions: pd.DataFrame = None,
                         sentiment_data: pd.DataFrame = None) -> Dict:
        """
        Backtest the trading strategy on historical data
        
        Args:
            historical_data: Historical price data
            technical_indicators: Historical technical indicators
            ml_predictions: Historical ML predictions (optional)
            sentiment_data: Historical sentiment data (optional)
            
        Returns:
            Backtest results
        """
        signals = []
        returns = []
        positions = []
        
        initial_capital = 10000
        current_position = 0
        cash = initial_capital
        portfolio_value = initial_capital
        
        for i in range(len(historical_data)):
            if i < 50:  # Skip initial period for indicator warmup
                continue
            
            current_data = historical_data.iloc[i]
            tech_indicators = technical_indicators.iloc[i].to_dict()
            
            ml_pred = ml_predictions.iloc[i].to_dict() if ml_predictions is not None else {}
            sentiment = sentiment_data.iloc[i].to_dict() if sentiment_data is not None else {}
            
            # Generate signal
            signal = self.generate_signal(
                current_data, tech_indicators, ml_pred, sentiment,
                historical_data.iloc[max(0, i-100):i]
            )
            
            signals.append(signal)
            
            # Execute trades (simplified)
            current_price = current_data['close']
            
            if signal.signal == Signal.BUY and current_position <= 0:
                # Buy signal
                shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash
                if shares_to_buy > 0:
                    current_position += shares_to_buy
                    cash -= shares_to_buy * current_price
            
            elif signal.signal == Signal.SELL and current_position > 0:
                # Sell signal
                cash += current_position * current_price
                current_position = 0
            
            # Calculate portfolio value
            portfolio_value = cash + current_position * current_price
            positions.append({
                'date': current_data.name,
                'portfolio_value': portfolio_value,
                'position': current_position,
                'cash': cash,
                'price': current_price
            })
            
            # Calculate return
            if i > 50:
                prev_value = positions[-2]['portfolio_value']
                period_return = (portfolio_value - prev_value) / prev_value
                returns.append(period_return)
        
        # Calculate performance metrics
        if returns:
            total_return = (portfolio_value - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            portfolio_values = [p['portfolio_value'] for p in positions]
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len([s for s in signals if s.signal != Signal.HOLD]),
                'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0,
                'positions': positions,
                'signals': signals
            }
        
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0,
            'win_rate': 0,
            'positions': [],
            'signals': []
        }