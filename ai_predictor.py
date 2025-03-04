import numpy as np
from sklearn.linear_model import LinearRegression

# Example AI model for stock prediction
def train_model(stock_data):
    # Dummy data: X = past prices, y = future prices
    X = np.array([data['past_price'] for data in stock_data]).reshape(-1, 1)
    y = np.array([data['future_price'] for data in stock_data])
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_stock_price(model, past_price):
    prediction = model.predict([[past_price]])
    return prediction[0]
