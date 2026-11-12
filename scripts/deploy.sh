#!/bin/bash

# Keke - Advanced AI-Powered Excel Datasheet Tool
# Comprehensive Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="keke-excel-tool"
NAMESPACE="keke"
# $1 is the command (see the case statement at the bottom); the environment is
# the second argument. Reading $1 here made `./deploy.sh deploy production` set
# ENVIRONMENT="deploy", so the production-only steps never ran.
ENVIRONMENT=${2:-"development"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"ghcr.io"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

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

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check if helm is installed (optional)
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed. Some features may not be available."
    fi
    
    log_success "Dependencies check completed"
}

setup_namespace() {
    log_info "Setting up namespace..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Add labels to namespace
    kubectl label namespace $NAMESPACE app=$PROJECT_NAME --overwrite
    kubectl label namespace $NAMESPACE environment=$ENVIRONMENT --overwrite
    
    log_success "Namespace setup completed"
}

deploy_configmaps() {
    log_info "Deploying ConfigMaps..."

    # Applied as committed. The previous version rewrote KEKE_ENV/DEBUG in place
    # with sed, which dirtied the working tree and never got reverted.
    kubectl apply -f k8s/configmap.yaml
    log_success "ConfigMaps deployed"
}

deploy_secrets() {
    log_info "Deploying Secrets..."

    # Secrets are provisioned out of band. The template is
    # k8s/secrets.yaml.example (the script previously looked for
    # k8s/secrets.yaml.template, which does not exist).
    if [ ! -f "k8s/secrets.yaml" ]; then
        log_error "k8s/secrets.yaml not found."
        log_error "cp k8s/secrets.yaml.example k8s/secrets.yaml, fill in the values, then re-run."
        exit 1
    fi

    kubectl apply -f k8s/secrets.yaml
    log_success "Secrets deployed"
}

deploy_storage() {
    log_info "Deploying Persistent Volume Claims..."
    
    kubectl apply -f k8s/pvc.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n $NAMESPACE --timeout=300s
    
    log_success "Storage deployed"
}

deploy_databases() {
    log_info "Deploying databases..."
    
    kubectl apply -f k8s/database-deployments.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=Available deployment/redis -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/mongodb -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/postgres -n $NAMESPACE --timeout=300s
    
    log_success "Databases deployed"
}

deploy_application() {
    log_info "Deploying application..."

    kubectl apply -f k8s/deployment.yaml

    # Set the image on the live Deployment rather than rewriting the manifest with
    # sed, which dirtied the repo and could not be re-run safely.
    log_info "Setting image to ${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}"
    kubectl set image deployment/keke-app \
        keke-app="${DOCKER_REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}" \
        -n $NAMESPACE

    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    kubectl wait --for=condition=Available deployment/keke-app -n $NAMESPACE --timeout=300s

    log_success "Application deployed"
}

deploy_services() {
    log_info "Deploying services..."
    
    kubectl apply -f k8s/service.yaml
    log_success "Services deployed"
}

deploy_monitoring() {
    log_info "Deploying monitoring..."
    
    kubectl apply -f k8s/monitoring.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring to be ready..."
    kubectl wait --for=condition=Available deployment/prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/grafana -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring deployed"
}

deploy_autoscaling() {
    log_info "Deploying autoscaling..."
    
    kubectl apply -f k8s/hpa.yaml
    log_success "Autoscaling deployed"
}

deploy_ingress() {
    log_info "Setting up ingress..."

    # The manifests still carry the placeholder hosts. Rewrite them in place only
    # when a real domain is supplied, so the repo is never silently modified.
    if [ -n "${INGRESS_HOST:-}" ]; then
        log_info "Setting ingress host to ${INGRESS_HOST}"
        sed -i.bak "s/keke\.yourdomain\.com/${INGRESS_HOST}/g; s/monitoring\.yourdomain\.com/monitoring.${INGRESS_HOST}/g" k8s/ingress.yaml
        mv k8s/ingress.yaml.bak k8s/ingress.yaml
    elif grep -q "keke.yourdomain.com" k8s/ingress.yaml; then
        log_warning "k8s/ingress.yaml still uses keke.yourdomain.com"
        log_warning "Edit the host, or re-run with INGRESS_HOST=your.domain"
    fi

    kubectl apply -f k8s/ingress.yaml
    log_success "Ingress deployed"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get service keke-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL="localhost:3000"
    fi
    
    # Wait for service to be ready
    log_info "Waiting for service to be ready..."
    sleep 30
    
    # Run health check
    if curl -f "http://$SERVICE_URL/health" > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Image: $DOCKER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG"
    echo ""
    
    log_info "Service URLs:"
    kubectl get services -n $NAMESPACE
    echo ""
    
    log_info "Pod Status:"
    kubectl get pods -n $NAMESPACE
    echo ""
    
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Ingress Information:"
        kubectl get ingress -n $NAMESPACE
        echo ""
    fi
    
    log_info "Monitoring URLs:"
    echo "Prometheus: http://localhost:9090 (port-forward)"
    echo "Grafana: http://localhost:3001 (port-forward)"
    echo ""
    
    log_info "To access the application:"
    echo "kubectl port-forward service/keke-service 3000:3000 -n $NAMESPACE"
    echo "Then visit: http://localhost:3000"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    # Remove any temporary files created during deployment
    rm -f k8s/*.tmp
}

# Main deployment function
deploy() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    
    check_dependencies
    setup_namespace
    deploy_configmaps
    deploy_secrets
    deploy_storage
    deploy_databases
    deploy_application
    deploy_services
    
    if [ "$ENVIRONMENT" = "production" ]; then
        deploy_monitoring
        deploy_autoscaling
        deploy_ingress
    fi
    
    run_health_checks
    show_deployment_info
    cleanup
    
    log_success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    kubectl rollout undo deployment/keke-app -n $NAMESPACE
    kubectl rollout status deployment/keke-app -n $NAMESPACE --timeout=300s
    
    log_success "Rollback completed"
}

# Status function
status() {
    log_info "Checking deployment status..."
    
    echo "Namespace: $NAMESPACE"
    echo "Environment: $ENVIRONMENT"
    echo ""
    
    kubectl get all -n $NAMESPACE
    echo ""
    
    kubectl get pvc -n $NAMESPACE
    echo ""
    
    kubectl get ingress -n $NAMESPACE
}

# Help function
show_help() {
    echo "Keke Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [ENVIRONMENT]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy the application (default)"
    echo "  rollback   Rollback the last deployment"
    echo "  status     Show deployment status"
    echo "  help       Show this help message"
    echo ""
    echo "Environments:"
    echo "  development  Deploy to development (default)"
    echo "  staging      Deploy to staging"
    echo "  production   Deploy to production"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_REGISTRY  Docker registry URL (default: ghcr.io)"
    echo "  IMAGE_TAG        Docker image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy development"
    echo "  $0 deploy production"
    echo "  $0 rollback"
    echo "  $0 status"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac