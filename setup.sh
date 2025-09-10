#!/bin/bash

echo "Setting up Keke Excel Datasheet Tool..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "Error: Node.js version 16+ is required. Current version: $(node -v)"
    exit 1
fi

echo "Installing Node.js dependencies..."
npm install

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p api/uploads
mkdir -p api/public

echo "Setting up file permissions..."
chmod +x api/excel_cli.py

echo "Setup complete!"
echo ""
echo "To start the server:"
echo "  npm start"
echo ""
echo "To start in development mode:"
echo "  npm run dev"
echo ""
echo "The web interface will be available at:"
echo "  http://localhost:3000"
echo ""
echo "Health check endpoint:"
echo "  http://localhost:3000/health"
