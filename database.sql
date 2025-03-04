CREATE TABLE stock_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    date DATE,
    open_price DECIMAL(10, 2),
    close_price DECIMAL(10, 2),
    high_price DECIMAL(10, 2),
    low_price DECIMAL(10, 2),
    volume INT
);

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    predicted_price DECIMAL(10, 2),
    prediction_date DATE
);
