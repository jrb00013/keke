import json
import requests # type: ignore

def lambda_handler(event, context):
    stock_symbol = event['queryStringParameters']['symbol']
    response = requests.get(f"https://api.example.com/stock/{stock_symbol}")
    stock_data = response.json()
    
    return {
        'statusCode': 200,
        'body': json.dumps(stock_data)
    }
