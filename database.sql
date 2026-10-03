-- Schema for the Keke stock-data tables.
--
-- This file is mounted into the PostgreSQL container
-- (/docker-entrypoint-initdb.d/init.sql), so it must be PostgreSQL syntax.
-- It previously used MySQL DDL (AUTO_INCREMENT), which aborts the Postgres
-- entrypoint (ON_ERROR_STOP=1) and leaves the database with no tables.

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS stock_data;

CREATE TABLE stock_data (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    date        DATE NOT NULL,
    open_price  NUMERIC(10, 2) NOT NULL,
    close_price NUMERIC(10, 2) NOT NULL,
    high_price  NUMERIC(10, 2) NOT NULL,
    low_price   NUMERIC(10, 2) NOT NULL,
    volume      INTEGER NOT NULL,
    UNIQUE (symbol, date) -- Ensures no duplicate stock entries per day
);

CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,
    predicted_price NUMERIC(10, 2) NOT NULL,
    prediction_date DATE NOT NULL,
    stock_id        INTEGER REFERENCES stock_data(id) ON DELETE CASCADE,
    UNIQUE (symbol, prediction_date) -- Prevents duplicate predictions for the same stock on the same day
);

CREATE INDEX idx_stock_symbol ON stock_data(symbol);
CREATE INDEX idx_prediction_symbol ON predictions(symbol);
