import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define stock symbols and the time period
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Example stock symbols
start_date = '2024-01-01'
end_date = '2025-01-01'

# Download historical data
def get_stock_data(symbols, start_date, end_date):
    stock_data = {}
    for symbol in symbols:
        stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

stock_data = get_stock_data(symbols, start_date, end_date)


def calculate_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                   data['Close'].diff(1).clip(upper=0).rolling(window=14).mean())))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    return data

# Apply technical indicators to all stock data
for symbol in stock_data:
    stock_data[symbol] = calculate_technical_indicators(stock_data[symbol])

def prepare_data_for_training(data, target_column='Close', look_back=5):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data.iloc[i-look_back:i].drop(columns=[target_column]))
        
        # Fixing the comparison by directly comparing the last value with the previous one
        if data[target_column].iloc[i] > data[target_column].iloc[i-1]:
            y.append(1)  # Buy
        else:
            y.append(0)  # Hold/Sell
    return np.array(X), np.array(y)



# Prepare training data for one stock (AAPL as an example)
X, y = prepare_data_for_training(stock_data['AAPL'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

def recommend_stocks(stock_data, model):
    recommendations = {}
    for symbol in stock_data:
        X, _ = prepare_data_for_training(stock_data[symbol])
        last_data = X[-1].reshape(1, -1)  # Take the last data point for prediction
        prediction = model.predict(last_data)
        recommendations[symbol] = 'Buy' if prediction[0] == 1 else 'Hold'
    return recommendations

# Get stock recommendations
recommendations = recommend_stocks(stock_data, model)
print("Stock Recommendations:")
for stock, recommendation in recommendations.items():
    print(f"{stock}: {recommendation}")



def plot_stock_data(stock_data, symbol):
    data = stock_data[symbol]
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA_50'], label='50-Day SMA')
    plt.plot(data['SMA_200'], label='200-Day SMA')
    plt.title(f'{symbol} Stock Price with Moving Averages')
    plt.legend()
    plt.show()

# Example: Plot stock data for AAPL
plot_stock_data(stock_data, 'AAPL')
