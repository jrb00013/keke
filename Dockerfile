# Multi-stage Docker build for Keke Advanced AI-Powered Excel Datasheet Tool

# Stage 1: Build Node.js application
FROM node:18-alpine AS node-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies including dev dependencies for build
RUN npm ci

# Copy source code
COPY api/ ./api/

# Build application if needed
RUN npm run build || true

# Stage 2: Build Python application with AI dependencies
FROM python:3.11-slim AS python-builder

WORKDIR /app

# Install system dependencies for AI libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY api/ ./api/

# Stage 3: Final production image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV PORT=3000
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgcc-s1 \
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
COPY run.py ./
COPY env.example ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/logs /app/data /app/temp /app/models /app/cache && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start command with proper error handling
CMD ["python3", "run.py", "start", "--mode", "app-only"]
