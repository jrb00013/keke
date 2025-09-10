#!/bin/bash

# Keke Excel Datasheet Tool - Comprehensive Deployment Script
# This script handles deployment to Docker, Kubernetes, and cloud platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="keke-excel-tool"
VERSION=${VERSION:-"latest"}
NAMESPACE=${NAMESPACE:-"keke"}
REGISTRY=${REGISTRY:-"registry.example.com"}
ENVIRONMENT=${ENVIRONMENT:-"development"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check kubectl for Kubernetes deployment
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm for Kubernetes deployment
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]] && ! command -v helm &> /dev/null; then
        log_warning "Helm is not installed, some features may not be available"
    fi
    
    log_success "Prerequisites check completed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    # Build the image
    docker build -t ${REGISTRY}/${PROJECT_NAME}:${VERSION} .
    
    if [[ $? -eq 0 ]]; then
        log_success "Docker image built successfully"
    else
        log_error "Docker image build failed"
        exit 1
    fi
    
    # Tag as latest if not already
    if [[ "$VERSION" != "latest" ]]; then
        docker tag ${REGISTRY}/${PROJECT_NAME}:${VERSION} ${REGISTRY}/${PROJECT_NAME}:latest
    fi
}

push_docker_image() {
    log_info "Pushing Docker image to registry..."
    
    docker push ${REGISTRY}/${PROJECT_NAME}:${VERSION}
    
    if [[ "$VERSION" != "latest" ]]; then
        docker push ${REGISTRY}/${PROJECT_NAME}:latest
    fi
    
    log_success "Docker image pushed successfully"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p uploads logs data ssl
    
    # Generate SSL certificates if they don't exist
    if [[ ! -f ssl/cert.pem ]]; then
        log_info "Generating self-signed SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    fi
    
    # Start services
    docker-compose up -d
    
    log_success "Docker Compose deployment completed"
    log_info "Services are starting up. Check status with: docker-compose ps"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply secrets
    kubectl apply -f k8s/keke-deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/keke-app -n ${NAMESPACE}
    
    log_success "Kubernetes deployment completed"
    log_info "Check status with: kubectl get pods -n ${NAMESPACE}"
}

run_tests() {
    log_info "Running comprehensive test suite..."
    
    # Run tests
    python run_tests.py --unit-only --integration-only
    
    if [[ $? -eq 0 ]]; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        # Start monitoring stack
        docker-compose up -d prometheus grafana elasticsearch logstash kibana
        
        log_success "Monitoring stack started"
        log_info "Access Grafana at: http://localhost:3001 (admin/admin)"
        log_info "Access Kibana at: http://localhost:5601"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Deploy monitoring to Kubernetes
        kubectl apply -f k8s/monitoring/
        
        log_success "Monitoring deployed to Kubernetes"
    fi
}

setup_backup() {
    log_info "Setting up backup strategy..."
    
    # Create backup script
    cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup databases
docker exec postgres pg_dump -U postgres keke > $BACKUP_DIR/postgres.sql
docker exec mongodb mongodump --db keke --out $BACKUP_DIR/mongodb

# Backup uploads
cp -r uploads $BACKUP_DIR/

# Backup logs
cp -r logs $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz -C /backups $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
EOF
    
    chmod +x backup.sh
    
    # Schedule backup cron job
    (crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/backup.sh") | crontab -
    
    log_success "Backup strategy configured"
}

health_check() {
    log_info "Performing health check..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        # Check Docker Compose services
        sleep 30
        curl -f http://localhost/health || {
            log_error "Health check failed"
            exit 1
        }
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Check Kubernetes services
        kubectl get pods -n ${NAMESPACE}
        kubectl get services -n ${NAMESPACE}
    fi
    
    log_success "Health check passed"
}

cleanup() {
    log_info "Cleaning up..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        docker-compose down
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        kubectl delete namespace ${NAMESPACE}
    fi
    
    log_success "Cleanup completed"
}

show_help() {
    echo "Keke Excel Datasheet Tool - Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE          Deployment type (docker-compose|kubernetes)"
    echo "  -e, --env ENVIRONMENT    Environment (development|staging|production)"
    echo "  -v, --version VERSION    Version tag (default: latest)"
    echo "  -r, --registry REGISTRY  Docker registry URL"
    echo "  -n, --namespace NAMESPACE Kubernetes namespace (default: keke)"
    echo "  --build-only             Only build Docker image"
    echo "  --push-only              Only push Docker image"
    echo "  --test-only              Only run tests"
    echo "  --monitoring-only        Only setup monitoring"
    echo "  --backup-only            Only setup backup"
    echo "  --health-check           Perform health check"
    echo "  --cleanup                Cleanup deployment"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --type docker-compose --env development"
    echo "  $0 --type kubernetes --env production --version v1.2.3"
    echo "  $0 --build-only --version v1.0.0"
}

# Parse command line arguments
DEPLOYMENT_TYPE="docker-compose"
BUILD_ONLY=false
PUSH_ONLY=false
TEST_ONLY=false
MONITORING_ONLY=false
BACKUP_ONLY=false
HEALTH_CHECK_ONLY=false
CLEANUP_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --push-only)
            PUSH_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --monitoring-only)
            MONITORING_ONLY=true
            shift
            ;;
        --backup-only)
            BACKUP_ONLY=true
            shift
            ;;
        --health-check)
            HEALTH_CHECK_ONLY=true
            shift
            ;;
        --cleanup)
            CLEANUP_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting Keke Excel Datasheet Tool deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    check_prerequisites
    
    if [[ "$BUILD_ONLY" == true ]]; then
        build_docker_image
        exit 0
    fi
    
    if [[ "$PUSH_ONLY" == true ]]; then
        push_docker_image
        exit 0
    fi
    
    if [[ "$TEST_ONLY" == true ]]; then
        run_tests
        exit 0
    fi
    
    if [[ "$MONITORING_ONLY" == true ]]; then
        setup_monitoring
        exit 0
    fi
    
    if [[ "$BACKUP_ONLY" == true ]]; then
        setup_backup
        exit 0
    fi
    
    if [[ "$HEALTH_CHECK_ONLY" == true ]]; then
        health_check
        exit 0
    fi
    
    if [[ "$CLEANUP_ONLY" == true ]]; then
        cleanup
        exit 0
    fi
    
    # Full deployment
    run_tests
    build_docker_image
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        push_docker_image
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        deploy_docker_compose
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        deploy_kubernetes
    fi
    
    setup_monitoring
    setup_backup
    health_check
    
    log_success "Deployment completed successfully!"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        log_info "Access the application at: http://localhost"
        log_info "Access Grafana at: http://localhost:3001"
        log_info "Access Kibana at: http://localhost:5601"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        log_info "Check deployment status: kubectl get pods -n $NAMESPACE"
        log_info "Access the application through your ingress controller"
    fi
}

# Run main function
main "$@"
