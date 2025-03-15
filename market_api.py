import requests
import time
import logging
import os

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO)

def fetch_stock_data(symbol, retries=3, delay=2):
    url = f"https://api.example.com/stock/{symbol}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            
            # Check if the response is successful
            if response.status_code == 200:
                logging.info(f"Successfully fetched data for {symbol}")
                return response.json()
            else:
                logging.error(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")
                return None
            
        except requests.exceptions.RequestException as e:
            # Handle any network-related errors
            logging.error(f"Error occurred while fetching data for {symbol}: {e}")
            
            # Retry logic
            if attempt < retries - 1:
                logging.info(f"Retrying ({attempt + 1}/{retries}) in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Max retries reached for {symbol}. Unable to fetch data.")
                return None
