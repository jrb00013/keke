-- Drop existing tables if they exist (optional)
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS stock_data;

-- Create stock_data table
CREATE TABLE stock_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10, 2) NOT NULL,
    close_price DECIMAL(10, 2) NOT NULL,
    high_price DECIMAL(10, 2) NOT NULL,
    low_price DECIMAL(10, 2) NOT NULL,
    volume INT NOT NULL,
    UNIQUE (symbol, date) -- Ensures no duplicate stock entries per day
);

-- Create predictions table
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    predicted_price DECIMAL(10, 2) NOT NULL,
    prediction_date DATE NOT NULL,
    stock_id INT,
    FOREIGN KEY (stock_id) REFERENCES stock_data(id) ON DELETE CASCADE,
    UNIQUE (symbol, prediction_date) -- Prevents duplicate predictions for the same stock on the same day
);

-- Indexing for faster queries
CREATE INDEX idx_stock_symbol ON stock_data(symbol);
CREATE INDEX idx_prediction_symbol ON predictions(symbol);
