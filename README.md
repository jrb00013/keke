# keke - A Specialized RTOS for Stock Market Integration and Datasheet Manipulation 📊

Implemented using Python, Cloud, AWS, and more.

## 🚀 Core Modules:
- **RTOS Kernel**:
  - `rtos_kernel.c` & `rtos_kernel.h`: Core logic.
  - `scheduler.c`: Manages tasks.
  - `task_manager.c`: Manages task creation.
  
- **Stock Market Integration**:
  - `market_api.py`: Fetches real-time stock data.
  - `ai_predictor.py`: Predicts stock trends.
  - `database.sql`: Stores stock data in an SQL database.

- **Cloud & API**:
  - `server.js`: Node.js backend.
  - `aws_lambda.py`: Handles AWS Lambda functions.
  
- **Deployment**:
  - `docker-compose.yml`: For containerization.
  - `requirements.txt`: Lists Python dependencies.

## 🚀 Quick Start:
1. Clone the repository.
2. Set up `.env` with your API keys.
3. Use `docker-compose up` to start the server.

## 📦 Requirements:
- Python 3.x
- Node.js
- Docker
- AWS Account
