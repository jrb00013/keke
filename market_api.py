import requests

def fetch_stock_data(symbol):
    url = f"https://api.example.com/stock/{symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
