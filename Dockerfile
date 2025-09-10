# Multi-stage Docker build for Keke Excel Datasheet Tool

# Stage 1: Build Node.js application
FROM node:18-alpine AS node-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY api/ ./api/

# Stage 2: Build Python application
FROM python:3.11-slim AS python-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY api/ ./api/

# Stage 3: Final production image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV PORT=3000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy Node.js dependencies from builder
COPY --from=node-builder /app/node_modules ./node_modules

# Copy application code
COPY api/ ./api/
COPY package*.json ./

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start command
CMD ["node", "api/server.js"]
