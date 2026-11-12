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

    # The application is the only default-profile service. Datastores,
    # monitoring, the proxy and ELK are opt-in profiles.
    if ! docker compose up -d --build keke-app; then
        log_error "docker compose failed to start keke-app"
        exit 1
    fi

    log_success "Docker Compose deployment completed"
    log_info "Services are starting up. Check status with: docker compose ps"
    log_info "Optional stacks: --profile databases | monitoring | proxy | logging"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    # Secrets are provisioned out of band (secrets.yaml.example is a template and
    # is never applied automatically).
    if [[ ! -f k8s/secrets.yaml ]]; then
        log_error "k8s/secrets.yaml is missing."
        log_error "Copy k8s/secrets.yaml.example, fill in the values, then re-run."
        exit 1
    fi

    # Apply in dependency order. k8s/deployment.yaml is the only Deployment for
    # the app; the previous k8s/keke-deployment.yaml was a duplicate with a
    # conflicting immutable selector and has been removed.
    for manifest in \
        k8s/namespace.yaml \
        k8s/secrets.yaml \
        k8s/configmap.yaml \
        k8s/pvc.yaml \
        k8s/database-deployments.yaml \
        k8s/deployment.yaml \
        k8s/service.yaml \
        k8s/ingress.yaml \
        k8s/hpa.yaml \
        k8s/monitoring.yaml
    do
        log_info "Applying ${manifest}"
        kubectl apply -f "${manifest}"
    done

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/keke-app -n ${NAMESPACE}

    log_success "Kubernetes deployment completed"
    log_info "Check status with: kubectl get pods -n ${NAMESPACE}"
}

run_tests() {
    log_info "Running test suite..."
    
    # Same commands CI runs. run_tests.py did not exist in the shipped tree and
    # --unit-only/--integration-only contradict each other.
    if ! npm test; then
        log_error "Node.js tests failed"
        exit 1
    fi
    
    if ! python3 -m pytest tests/ -q; then
        log_error "Python tests failed"
        exit 1
    fi
    
    log_success "All tests passed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."

    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        # Prometheus/Grafana are behind the monitoring profile.
        docker compose --profile monitoring up -d prometheus grafana

        log_success "Monitoring stack started"
        log_info "Access Grafana at: http://localhost:3001 (admin/admin)"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # The manifests live in a single file, not a directory.
        kubectl apply -f k8s/monitoring.yaml

        log_success "Monitoring deployed to Kubernetes"
    fi
}

setup_backup() {
    log_info "Setting up backup strategy..."

    cat > backup.sh << 'EOF'
#!/bin/bash
# Back up the datastores and session data.
# Requires the datastores to be running:
#   docker compose --profile databases up -d
set -euo pipefail

BACKUP_ROOT="${BACKUP_ROOT:-./backups}"
STAMP=$(date +%Y%m%d_%H%M%S)
DEST="${BACKUP_ROOT}/${STAMP}"
mkdir -p "$DEST"

# Resolve the real container names instead of hard-coding 'postgres'/'mongodb'.
POSTGRES_CONTAINER=$(docker compose ps -q postgres 2>/dev/null || true)
MONGO_CONTAINER=$(docker compose ps -q mongodb 2>/dev/null || true)

if [[ -n "$POSTGRES_CONTAINER" ]]; then
    docker exec "$POSTGRES_CONTAINER" pg_dumpall -U postgres > "$DEST/postgres.sql"
    echo "Backed up PostgreSQL"
else
    echo "WARNING: postgres container not running; skipping"
fi

if [[ -n "$MONGO_CONTAINER" ]]; then
    docker exec "$MONGO_CONTAINER" mongodump --db keke --out /tmp/keke-dump >/dev/null
    docker cp "$MONGO_CONTAINER":/tmp/keke-dump "$DEST/mongodb"
    echo "Backed up MongoDB"
else
    echo "WARNING: mongodb container not running; skipping"
fi

# Session data is the application's actual state.
if [[ -d data/sessions ]]; then
    cp -r data/sessions "$DEST/sessions"
    echo "Backed up session data"
else
    echo "WARNING: data/sessions not found; skipping"
fi

tar -czf "$DEST.tar.gz" -C "$BACKUP_ROOT" "$STAMP"
rm -rf "$DEST"
echo "Backup completed: $DEST.tar.gz"
EOF

    chmod +x backup.sh

    # Schedule backup cron job
    (crontab -l 2>/dev/null; echo "0 2 * * * cd $(pwd) && ./backup.sh") | crontab -

    log_success "Backup strategy configured"
}

health_check() {
    log_info "Performing health check..."

    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        # Give the container time to come up before probing.
        sleep 30
        # The app publishes 3000 directly; the nginx proxy is an opt-in profile.
        curl -f "http://localhost:${APP_PORT:-3000}/health" || {
            log_error "Health check failed"
            exit 1
        }
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Check Kubernetes services
        kubectl get pods -n ${NAMESPACE}
        kubectl get services -n ${NAMESPACE}
        kubectl port-forward service/keke-service "${APP_PORT:-3000}:3000" -n ${NAMESPACE} &
        PF_PID=$!
        sleep 5
        curl -f "http://localhost:${APP_PORT:-3000}/health" || {
            kill "$PF_PID" 2>/dev/null || true
            log_error "Health check failed"
            exit 1
        }
        kill "$PF_PID" 2>/dev/null || true
    fi

    log_success "Health check passed"
}

cleanup() {
    log_info "Cleaning up..."

    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        docker compose down --remove-orphans
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
        log_info "Access the application at: http://localhost:${APP_PORT:-3000}"
        log_info "Access Grafana at: http://localhost:3001"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        log_info "Check deployment status: kubectl get pods -n $NAMESPACE"
        log_info "Access the application through your ingress controller"
    fi
}

# Run main function
main "$@"
