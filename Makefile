# Makefile for Testing Observability Platform
# Handles both Docker Compose and Kubernetes deployment

# Configuration
APP_NAME := testing-observability-platform
DOCKER_REGISTRY := viteadevacr.azurecr.io

# Image names
API_IMAGE_LOCAL := evaluator-api-local
WORKER_IMAGE_LOCAL := evaluator-worker-local
EVALUATOR_IMAGE_LOCAL := evaluator-service-local

API_IMAGE_CLOUD := $(DOCKER_REGISTRY)/evaluator-api
WORKER_IMAGE_CLOUD := $(DOCKER_REGISTRY)/evaluator-worker
EVALUATOR_IMAGE_CLOUD := $(DOCKER_REGISTRY)/evaluator-service

# Default environment
ENV ?= local

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  Docker Compose:"
	@echo "    dc-build       - Build Docker Compose images"
	@echo "    dc-up          - Start Docker Compose services"
	@echo "    dc-down        - Stop Docker Compose services"
	@echo ""
	@echo "  Kubernetes:"
	@echo "    k8s-build      - Build Kubernetes images for local deployment"
	@echo "    k8s-deploy     - Deploy to Kubernetes (local by default)"
	@echo "    k8s-delete     - Delete from Kubernetes"
	@echo "    k8s-status     - Show Kubernetes deployment status"
	@echo ""
	@echo "  Environment: ENV=local|dev|staging|prod (default: local)"

# ============================================================================
# Docker Compose Commands
# ============================================================================

.PHONY: dc-build
dc-build:
	docker-compose build

.PHONY: dc-up
dc-up:
	docker-compose up -d

.PHONY: dc-down
dc-down:
	docker-compose down

.PHONY: dc-logs
dc-logs:
	docker-compose logs -f

# ============================================================================
# Kubernetes Commands
# ============================================================================

.PHONY: k8s-build
k8s-build:
ifeq ($(ENV),local)
	@echo "Building local Kubernetes images..."
	# Build API image
	docker build -t $(API_IMAGE_LOCAL):latest .
	# Worker uses same image as API
	docker tag $(API_IMAGE_LOCAL):latest $(WORKER_IMAGE_LOCAL):latest
	# Build evaluator service (assumes ../vitea-evaluators exists)
	@if [ -d "../vitea-evaluators" ]; then \
		echo "Building evaluator service image..."; \
		docker build -t $(EVALUATOR_IMAGE_LOCAL):latest ../vitea-evaluators; \
	else \
		echo "Warning: ../vitea-evaluators not found, skipping evaluator image build"; \
	fi
	@echo "Local Kubernetes images built successfully!"
	@echo "Images created:"
	@echo "  - $(API_IMAGE_LOCAL):latest"
	@echo "  - $(WORKER_IMAGE_LOCAL):latest"
	@echo "  - $(EVALUATOR_IMAGE_LOCAL):latest (if vitea-evaluators available)"
else
	@echo "Building cloud images for $(ENV)..."
	# Build and push API image
	docker build -t $(API_IMAGE_CLOUD):latest .
	docker push $(API_IMAGE_CLOUD):latest
	# Worker uses same image as API
	docker tag $(API_IMAGE_CLOUD):latest $(WORKER_IMAGE_CLOUD):latest
	docker push $(WORKER_IMAGE_CLOUD):latest
	# Build evaluator service
	@if [ -d "../vitea-evaluators" ]; then \
		docker build -t $(EVALUATOR_IMAGE_CLOUD):latest ../vitea-evaluators; \
		docker push $(EVALUATOR_IMAGE_CLOUD):latest; \
	fi
	@echo "Cloud images built and pushed successfully!"
endif

.PHONY: k8s-deploy
k8s-deploy:
ifeq ($(ENV),local)
	@echo "Deploying to local Kubernetes (Minikube)..."
	helm upgrade --install evaluator ./helm \
		-f ./helm/values-local.yaml \
		--namespace vitea-data \
		--create-namespace
else
	@echo "Deploying to $(ENV) environment..."
	helm upgrade --install evaluator ./helm \
		-f ./helm/values.yaml \
		--namespace vitea-data \
		--create-namespace \
		--set env.APP_ENV=$(ENV)
endif
	@echo "Deployment completed!"
	@echo ""
	@echo "Check status with: make k8s-status"
	@echo "Access via: https://eval.local.vitea.ai (local) or https://eval.$(ENV).vitea.ai"

.PHONY: k8s-delete
k8s-delete:
	helm uninstall evaluator --namespace vitea-data
	@echo "Kubernetes deployment deleted!"

.PHONY: k8s-status
k8s-status:
	@echo "=== Helm Release Status ==="
	helm status evaluator --namespace vitea-data
	@echo ""
	@echo "=== Pod Status ==="
	kubectl get pods -n vitea-data -l app.kubernetes.io/instance=evaluator
	@echo ""
	@echo "=== Service Status ==="
	kubectl get services -n vitea-data -l app.kubernetes.io/instance=evaluator

.PHONY: k8s-logs
k8s-logs:
	@echo "Choose component to view logs:"
	@echo "1. API:       kubectl logs -n vitea-data -l app.kubernetes.io/component=api"
	@echo "2. Worker:    kubectl logs -n vitea-data -l app.kubernetes.io/component=worker"
	@echo "3. Evaluator: kubectl logs -n vitea-data -l app.kubernetes.io/component=evaluator"

.PHONY: k8s-shell
k8s-shell:
	kubectl exec -it -n vitea-data deployment/evaluator-api -- /bin/bash

# ============================================================================
# Development Commands
# ============================================================================

.PHONY: format
format:
	uv run black app/ tests/
	uv run isort app/ tests/

.PHONY: lint
lint:
	uv run ruff check app/ tests/

.PHONY: test
test:
	uv run pytest

.PHONY: migrate
migrate:
	uv run alembic upgrade head

# ============================================================================
# Combined Commands
# ============================================================================

.PHONY: build-all
build-all: dc-build k8s-build

.PHONY: local-deploy
local-deploy: k8s-build k8s-deploy

.PHONY: local-redeploy
local-redeploy: k8s-build k8s-delete k8s-deploy

# ============================================================================
# Validation Commands
# ============================================================================

.PHONY: k8s-validate
k8s-validate:
	@echo "Validating Helm templates..."
	helm template evaluator ./helm -f ./helm/values-local.yaml --namespace vitea-data

.PHONY: k8s-dry-run
k8s-dry-run:
	@echo "Performing dry-run deployment..."
	helm upgrade --install evaluator ./helm \
		-f ./helm/values-local.yaml \
		--namespace vitea-data \
		--create-namespace \
		--dry-run --debug