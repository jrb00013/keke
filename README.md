# keke - A Specialized RTOS for Datasheet Manipulation 

Implemented using Python, C, Node.js, and AWS Lambda.

## Core Modules:
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

## Requirements:
- Python 3.x
- Node.js
- Docker
- AWS Account
