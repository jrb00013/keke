#!/bin/bash

echo "🚀 Deploying keke RTOS & Stock Market Integration..."

# Step 1: Build RTOS Kernel
echo "🔹 Building RTOS Kernel..."
make clean && make

# Step 2: Start the backend server
echo "🔹 Starting Backend Server..."
nohup node src/backend/server.js > logs/server.log 2>&1 &

# Step 3: Start database setup
echo "🔹 Setting Up Database..."
python3 scripts/setup_db.py

# Step 4: Start Stock Market Integration Services
echo "🔹 Starting Stock Market API Service..."
nohup python3 src/
