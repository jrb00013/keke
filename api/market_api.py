import requests
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"

@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open_price: float
    timestamp: datetime
    market_status: MarketStatus

class MarketDataProvider:
    """
    Enhanced market data provider with multiple API sources and fallback mechanisms
    """
    
    def __init__(self):
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY')
        }
        
        self.base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1',
            'polygon': 'https://api.polygon.io/v2',
            'yahoo': 'https://query1.finance.yahoo.com/v8/finance/chart'
        }
        
        self.rate_limits = {
            'alpha_vantage': 5,  # requests per minute
            'finnhub': 60,       # requests per minute
            'polygon': 5         # requests per minute
        }
        
        self.last_request_times = {}
    
    def _check_rate_limit(self, provider: str) -> bool:
        """Check if we can make a request without hitting rate limits"""
        if provider not in self.last_request_times:
            return True
        
        last_time = self.last_request_times[provider]
        time_diff = time.time() - last_time
        min_interval = 60 / self.rate_limits[provider]
        
        return time_diff >= min_interval
    
    def _update_rate_limit(self, provider: str):
        """Update the last request time for rate limiting"""
        self.last_request_times[provider] = time.time()
    
    def fetch_stock_data(self, symbol: str, retries: int = 3, delay: int = 2) -> Optional[StockData]:
        """
        Fetch stock data with multiple provider fallback
        """
        symbol = symbol.upper()
        
        # Try different providers in order of preference
        providers = ['alpha_vantage', 'finnhub', 'polygon', 'yahoo']
        
        for provider in providers:
            if not self._check_rate_limit(provider):
                logger.warning(f"Rate limit reached for {provider}, skipping")
                continue
            
            for attempt in range(retries):
                try:
                    data = self._fetch_from_provider(symbol, provider)
                    if data:
                        self._update_rate_limit(provider)
                        logger.info(f"Successfully fetched data for {symbol} from {provider}")
                        return data
                    
                except Exception as e:
                    logger.error(f"Error fetching from {provider} (attempt {attempt + 1}): {e}")
                    
                    if attempt < retries - 1:
                        logger.info(f"Retrying {provider} in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for {provider}")
                        break
        
        logger.error(f"Failed to fetch data for {symbol} from all providers")
        return None
    
    def _fetch_from_provider(self, symbol: str, provider: str) -> Optional[StockData]:
        """Fetch data from a specific provider"""
        
        if provider == 'alpha_vantage':
            return self._fetch_alpha_vantage(symbol)
        elif provider == 'finnhub':
            return self._fetch_finnhub(symbol)
        elif provider == 'polygon':
            return self._fetch_polygon(symbol)
        elif provider == 'yahoo':
            return self._fetch_yahoo(symbol)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _fetch_alpha_vantage(self, symbol: str) -> Optional[StockData]:
        """Fetch data from Alpha Vantage API"""
        if not self.api_keys['alpha_vantage']:
            return None
        
        url = self.base_urls['alpha_vantage']
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_keys['alpha_vantage']
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Global Quote' not in data:
            return None
        
        quote = data['Global Quote']
        
        return StockData(
            symbol=symbol,
            price=float(quote['05. price']),
            change=float(quote['09. change']),
            change_percent=float(quote['10. change percent'].rstrip('%')),
            volume=int(quote['06. volume']),
            high=float(quote['03. high']),
            low=float(quote['04. low']),
            open_price=float(quote['02. open']),
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN
        )
    
    def _fetch_finnhub(self, symbol: str) -> Optional[StockData]:
        """Fetch data from Finnhub API"""
        if not self.api_keys['finnhub']:
            return None
        
        url = f"{self.base_urls['finnhub']}/quote"
        params = {
            'symbol': symbol,
            'token': self.api_keys['finnhub']
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'c' not in data:  # 'c' is current price
            return None
        
        return StockData(
            symbol=symbol,
            price=data['c'],
            change=data['d'],
            change_percent=data['dp'],
            volume=data.get('v', 0),
            high=data.get('h', data['c']),
            low=data.get('l', data['c']),
            open_price=data.get('o', data['c']),
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN
        )
    
    def _fetch_polygon(self, symbol: str) -> Optional[StockData]:
        """Fetch data from Polygon API"""
        if not self.api_keys['polygon']:
            return None
        
        url = f"{self.base_urls['polygon']}/aggs/ticker/{symbol}/prev"
        params = {
            'apikey': self.api_keys['polygon']
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data or not data['results']:
            return None
        
        result = data['results'][0]
        
        return StockData(
            symbol=symbol,
            price=result['c'],
            change=result['c'] - result['o'],
            change_percent=((result['c'] - result['o']) / result['o']) * 100,
            volume=result['v'],
            high=result['h'],
            low=result['l'],
            open_price=result['o'],
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN
        )
    
    def _fetch_yahoo(self, symbol: str) -> Optional[StockData]:
        """Fetch data from Yahoo Finance (no API key required)"""
        url = f"{self.base_urls['yahoo']}/{symbol}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'chart' not in data or not data['chart']['result']:
            return None
        
        result = data['chart']['result'][0]
        meta = result['meta']
        quote = result['indicators']['quote'][0]
        
        current_price = meta['regularMarketPrice']
        previous_close = meta['previousClose']
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        return StockData(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=meta.get('regularMarketVolume', 0),
            high=meta.get('regularMarketDayHigh', current_price),
            low=meta.get('regularMarketDayLow', current_price),
            open_price=meta.get('regularMarketOpen', current_price),
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN
        )
    
    async def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Fetch data for multiple stocks concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_stock_async(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                symbol: result if not isinstance(result, Exception) else None
                for symbol, result in zip(symbols, results)
            }
    
    async def _fetch_stock_async(self, session: aiohttp.ClientSession, symbol: str) -> Optional[StockData]:
        """Async version of stock data fetching"""
        try:
            # Use Yahoo Finance for async requests (no rate limits)
            url = f"{self.base_urls['yahoo']}/{symbol}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_data(symbol, data)
                else:
                    logger.error(f"HTTP {response.status} for {symbol}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def _parse_yahoo_data(self, symbol: str, data: dict) -> Optional[StockData]:
        """Parse Yahoo Finance data"""
        if 'chart' not in data or not data['chart']['result']:
            return None
        
        result = data['chart']['result'][0]
        meta = result['meta']
        
        current_price = meta['regularMarketPrice']
        previous_close = meta['previousClose']
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        return StockData(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=meta.get('regularMarketVolume', 0),
            high=meta.get('regularMarketDayHigh', current_price),
            low=meta.get('regularMarketDayLow', current_price),
            open_price=meta.get('regularMarketOpen', current_price),
            timestamp=datetime.now(),
            market_status=MarketStatus.OPEN
        )

# Global instance
market_provider = MarketDataProvider()

# Convenience functions for backward compatibility
def fetch_stock_data(symbol, retries=3, delay=2):
    """Legacy function for backward compatibility"""
    stock_data = market_provider.fetch_stock_data(symbol, retries, delay)
    if stock_data:
        return {
            'symbol': stock_data.symbol,
            'price': stock_data.price,
            'change': stock_data.change,
            'change_percent': stock_data.change_percent,
            'volume': stock_data.volume,
            'high': stock_data.high,
            'low': stock_data.low,
            'open': stock_data.open_price,
            'timestamp': stock_data.timestamp.isoformat()
        }
    return None
