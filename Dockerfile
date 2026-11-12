# Multi-stage Docker build for Keke Advanced AI-Powered Excel Datasheet Tool

# Stage 1: Build Node.js application
#
# Debian-based rather than Alpine on purpose. Stage 3 is python:3.11-slim
# (glibc), so the node binary copied out of this stage has to be glibc-linked;
# a musl/alpine binary would not run there.
FROM node:18-bookworm-slim AS node-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies including dev dependencies for build
RUN npm ci

# Copy source code
COPY api/ ./api/

# Stage 2: Build Python application with AI dependencies
#
# No compiler or BLAS/ATLAS headers are installed: numpy, pandas, scipy,
# scikit-learn, pyarrow and matplotlib all publish manylinux wheels for
# cp311, so pip never needs to compile anything. The previous version of this
# stage installed a full toolchain including libatlas-base-dev and
# libgl1-mesa-glx, both of which have been removed from Debian. That made the
# image unbuildable: apt failed with "no installation candidate" long before
# pip ever ran.
FROM python:3.11-slim AS python-builder

WORKDIR /app

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

# Install system dependencies.
#
# curl is for HEALTHCHECK, libgomp1 is the OpenMP runtime that the
# scikit-learn/scipy wheels link against. The GUI libraries that used to be
# here (libsm6, libxext6, libxrender-dev, libgl1-mesa-glx) are not needed
# because matplotlib runs headless under MPLBACKEND=Agg. node and npm are
# taken from stage 1 instead of Debian's nodejs/npm packages so the runtime
# always matches the version node_modules was built with.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy the Node runtime and dependencies from builder
COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=node-builder /app/node_modules ./node_modules

# Copy application code
COPY api/ ./api/
# legacy/ holds freertos_integration.py, which api/rtos_bridge.py imports at
# module load. Without it every /api/rtos/* route fails with ImportError in
# the image. The C sources under legacy/rtos/ are reference material only and
# stay out of the image via .dockerignore.
COPY legacy/ ./legacy/
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
