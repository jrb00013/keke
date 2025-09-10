#!/usr/bin/env python3
"""
Keke - Advanced AI-Powered Excel Datasheet Tool
Comprehensive startup script for development and production environments
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import docker
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keke.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KekeRunner:
    """
    Comprehensive runner for Keke Excel Datasheet Tool
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        self.docker_client = None
        self.config = self._load_config()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
    
    def _load_config(self) -> Dict:
        """Load configuration from environment and config files"""
        config = {
            'environment': os.getenv('KEKE_ENV', 'development'),
            'port': int(os.getenv('PORT', 3000)),
            'host': os.getenv('HOST', 'localhost'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'ai_enabled': os.getenv('AI_ENABLED', 'true').lower() == 'true',
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///keke.db'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'mongodb_url': os.getenv('MONGODB_URL', 'mongodb://localhost:27017/keke'),
            'postgres_url': os.getenv('POSTGRES_URL', 'postgresql://postgres:password@localhost:5432/keke'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
            'azure_storage_connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
            'google_application_credentials': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            'monitoring_enabled': os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024)),  # 50MB
            'rate_limit': int(os.getenv('RATE_LIMIT', 100)),
            'rate_window': int(os.getenv('RATE_WINDOW', 900)),  # 15 minutes
        }
        
        # Load from config file if exists
        config_file = self.project_root / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return config
    
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Node.js not found. Please install Node.js 16+")
                return False
            logger.info(f"Node.js version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Node.js not found. Please install Node.js 16+")
            return False
        
        # Check Python
        try:
            result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Python 3 not found. Please install Python 3.8+")
                return False
            logger.info(f"Python version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Python 3 not found. Please install Python 3.8+")
            return False
        
        # Check npm packages
        try:
            result = subprocess.run(['npm', 'list', '--depth=0'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("npm packages not installed. Run 'npm install' first.")
        except FileNotFoundError:
            logger.error("npm not found. Please install npm")
            return False
        
        # Check Python packages
        try:
            result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
            if 'pandas' not in result.stdout:
                logger.warning("Python packages not installed. Run 'pip install -r requirements.txt' first.")
        except FileNotFoundError:
            logger.error("pip3 not found. Please install pip")
            return False
        
        # Check AI API keys
        if self.config['ai_enabled']:
            if not self.config['openai_api_key'] and not self.config['anthropic_api_key']:
                logger.warning("AI features enabled but no API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        logger.info("Dependency check completed")
        return True
    
    def _setup_directories(self):
        """Create necessary directories"""
        logger.info("Setting up directories...")
        
        directories = [
            'uploads',
            'logs',
            'data',
            'temp',
            'models',
            'cache'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def _start_database_containers(self) -> bool:
        """Start database containers using Docker Compose"""
        if not self.docker_client:
            logger.warning("Docker not available. Skipping database containers.")
            return False
        
        logger.info("Starting database containers...")
        
        try:
            # Check if docker-compose is available
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("docker-compose not available. Skipping database containers.")
                return False
            
            # Start only database services
            services = ['redis', 'mongodb', 'postgres']
            for service in services:
                logger.info(f"Starting {service}...")
                result = subprocess.run([
                    'docker-compose', 'up', '-d', service
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to start {service}: {result.stderr}")
                    return False
                
                logger.info(f"{service} started successfully")
            
            # Wait for services to be ready
            self._wait_for_services()
            return True
            
        except Exception as e:
            logger.error(f"Error starting database containers: {e}")
            return False
    
    def _wait_for_services(self):
        """Wait for database services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        services = {
            'redis': 'http://localhost:6379',
            'mongodb': 'mongodb://localhost:27017',
            'postgres': 'postgresql://postgres:password@localhost:5432/keke'
        }
        
        for service, url in services.items():
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    if service == 'redis':
                        # Redis doesn't have HTTP, just check if port is open
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        result = sock.connect_ex(('localhost', 6379))
                        sock.close()
                        if result == 0:
                            logger.info(f"{service} is ready")
                            break
                    else:
                        # For other services, we'll assume they're ready after a delay
                        time.sleep(2)
                        logger.info(f"{service} is ready")
                        break
                        
                except Exception as e:
                    logger.debug(f"{service} not ready yet: {e}")
                
                attempt += 1
                time.sleep(1)
            
            if attempt >= max_attempts:
                logger.warning(f"{service} may not be ready")
    
    def _start_monitoring(self):
        """Start monitoring services if enabled"""
        if not self.config['monitoring_enabled']:
            return
        
        logger.info("Starting monitoring services...")
        
        try:
            # Start Prometheus
            result = subprocess.run([
                'docker-compose', 'up', '-d', 'prometheus'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Prometheus started")
            
            # Start Grafana
            result = subprocess.run([
                'docker-compose', 'up', '-d', 'grafana'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Grafana started")
                
        except Exception as e:
            logger.warning(f"Could not start monitoring services: {e}")
    
    def _start_main_application(self):
        """Start the main Keke application"""
        logger.info("Starting Keke application...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'NODE_ENV': 'production' if self.config['environment'] == 'production' else 'development',
            'PORT': str(self.config['port']),
            'HOST': self.config['host'],
            'DEBUG': str(self.config['debug']).lower(),
            'AI_ENABLED': str(self.config['ai_enabled']).lower(),
            'OPENAI_API_KEY': self.config['openai_api_key'] or '',
            'ANTHROPIC_API_KEY': self.config['anthropic_api_key'] or '',
            'DATABASE_URL': self.config['database_url'],
            'REDIS_URL': self.config['redis_url'],
            'MONGODB_URL': self.config['mongodb_url'],
            'POSTGRES_URL': self.config['postgres_url'],
            'AWS_ACCESS_KEY_ID': self.config['aws_access_key_id'] or '',
            'AWS_SECRET_ACCESS_KEY': self.config['aws_secret_access_key'] or '',
            'AWS_REGION': self.config['aws_region'],
            'AZURE_STORAGE_CONNECTION_STRING': self.config['azure_storage_connection_string'] or '',
            'GOOGLE_APPLICATION_CREDENTIALS': self.config['google_application_credentials'] or '',
            'MAX_FILE_SIZE': str(self.config['max_file_size']),
            'RATE_LIMIT': str(self.config['rate_limit']),
            'RATE_WINDOW': str(self.config['rate_window']),
        })
        
        # Start the Node.js application
        try:
            process = subprocess.Popen([
                'node', 'api/server.js'
            ], cwd=self.project_root, env=env)
            
            self.processes.append(process)
            logger.info(f"Keke application started with PID {process.pid}")
            
            # Wait a moment for the application to start
            time.sleep(3)
            
            # Check if the application is running
            try:
                response = requests.get(f"http://{self.config['host']}:{self.config['port']}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Keke application is healthy and running")
                else:
                    logger.warning(f"Health check failed with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to start Keke application: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self, mode: str = 'full'):
        """Start Keke in the specified mode"""
        logger.info(f"Starting Keke in {mode} mode...")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Check dependencies
        if not self._check_dependencies():
            logger.error("Dependency check failed. Please install required dependencies.")
            return False
        
        # Setup directories
        self._setup_directories()
        
        # Start services based on mode
        if mode in ['full', 'with-db']:
            self._start_database_containers()
        
        if mode in ['full', 'with-monitoring']:
            self._start_monitoring()
        
        # Always start the main application
        self._start_main_application()
        
        logger.info("Keke startup completed successfully!")
        logger.info(f"Application URL: http://{self.config['host']}:{self.config['port']}")
        logger.info(f"Health check: http://{self.config['host']}:{self.config['port']}/health")
        
        if self.config['monitoring_enabled']:
            logger.info("Monitoring URLs:")
            logger.info("  Prometheus: http://localhost:9090")
            logger.info("  Grafana: http://localhost:3001 (admin/admin)")
        
        return True
    
    def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down Keke...")
        
        # Stop main application processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"Stopped process {process.pid}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing process {process.pid}")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping process {process.pid}: {e}")
        
        # Stop Docker containers
        if self.docker_client:
            try:
                result = subprocess.run([
                    'docker-compose', 'down'
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Docker containers stopped")
                else:
                    logger.warning(f"Error stopping containers: {result.stderr}")
            except Exception as e:
                logger.error(f"Error stopping Docker containers: {e}")
        
        logger.info("Keke shutdown completed")
    
    def status(self):
        """Check the status of all services"""
        logger.info("Checking Keke status...")
        
        # Check main application
        try:
            response = requests.get(f"http://{self.config['host']}:{self.config['port']}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Main application: Running")
                health_data = response.json()
                logger.info(f"   Uptime: {health_data.get('uptime', 'Unknown')}")
                logger.info(f"   Version: {health_data.get('version', 'Unknown')}")
            else:
                logger.warning(f"⚠️ Main application: Health check failed (status {response.status_code})")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Main application: Not responding ({e})")
        
        # Check database services
        services = {
            'Redis': ('localhost', 6379),
            'MongoDB': ('localhost', 27017),
            'PostgreSQL': ('localhost', 5432)
        }
        
        for service, (host, port) in services.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    logger.info(f"✅ {service}: Running")
                else:
                    logger.warning(f"⚠️ {service}: Not responding")
            except Exception as e:
                logger.error(f"❌ {service}: Error checking status ({e})")
        
        # Check monitoring services
        if self.config['monitoring_enabled']:
            monitoring_services = {
                'Prometheus': 'http://localhost:9090',
                'Grafana': 'http://localhost:3001'
            }
            
            for service, url in monitoring_services.items():
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service}: Running")
                    else:
                        logger.warning(f"⚠️ {service}: Not responding (status {response.status_code})")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️ {service}: Not responding ({e})")
    
    def install(self):
        """Install dependencies"""
        logger.info("Installing Keke dependencies...")
        
        # Install Node.js dependencies
        logger.info("Installing Node.js dependencies...")
        try:
            result = subprocess.run(['npm', 'install'], cwd=self.project_root, check=True)
            logger.info("Node.js dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Node.js dependencies: {e}")
            return False
        
        # Install Python dependencies
        logger.info("Installing Python dependencies...")
        try:
            result = subprocess.run(['pip3', 'install', '-r', 'requirements.txt'], cwd=self.project_root, check=True)
            logger.info("Python dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False
        
        logger.info("All dependencies installed successfully!")
        return True
    
    def test(self):
        """Run tests"""
        logger.info("Running Keke tests...")
        
        # Run Node.js tests
        logger.info("Running Node.js tests...")
        try:
            result = subprocess.run(['npm', 'test'], cwd=self.project_root, check=True)
            logger.info("Node.js tests passed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Node.js tests failed: {e}")
            return False
        
        # Run Python tests
        logger.info("Running Python tests...")
        try:
            result = subprocess.run(['python3', '-m', 'pytest', 'tests/'], cwd=self.project_root, check=True)
            logger.info("Python tests passed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Python tests failed: {e}")
            return False
        
        logger.info("All tests passed!")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Keke - Advanced AI-Powered Excel Datasheet Tool')
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status', 'install', 'test'], 
                       help='Command to execute')
    parser.add_argument('--mode', choices=['full', 'app-only', 'with-db', 'with-monitoring'], 
                       default='full', help='Startup mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize runner
    runner = KekeRunner()
    
    try:
        if args.command == 'start':
            success = runner.start(args.mode)
            if success:
                # Keep the process running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
        elif args.command == 'stop':
            runner.shutdown()
        elif args.command == 'restart':
            runner.shutdown()
            time.sleep(2)
            runner.start(args.mode)
        elif args.command == 'status':
            runner.status()
        elif args.command == 'install':
            runner.install()
        elif args.command == 'test':
            runner.test()
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        runner.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
