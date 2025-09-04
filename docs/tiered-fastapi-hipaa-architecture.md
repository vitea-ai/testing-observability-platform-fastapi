# Tiered FastAPI HIPAA Architecture: Progressive Enhancement Strategy

## Overview: 4-Tier Progressive Implementation

This guide provides a pragmatic approach to building HIPAA-compliant FastAPI applications through progressive enhancement. Start with a simple scaffold for rapid development, then systematically add security, performance, and compliance features as you move toward production.

```mermaid
graph LR
    A[Tier 1: Local Dev] --> B[Tier 2: Integration]
    B --> C[Tier 3: Staging/Dev Deploy]
    C --> D[Tier 4: Production]
    
    A -.-> |Simple FastAPI| A1[Business Logic Focus]
    B -.-> |+Auth & DB| B1[Team Testing]
    C -.-> |+Security & Monitoring| C1[UAT Ready]
    D -.-> |+Full Compliance| D1[HIPAA Compliant]
```

## Tier 1: Local Development (Day 1)

### Goals
- Rapid prototyping and business logic development
- Minimal dependencies and configuration
- Fast iteration cycles
- Mock external services

### Project Structure

```bash
# Initialize with uv
uv init governance-platform
cd governance-platform
uv add fastapi uvicorn pydantic sqlalchemy
uv add --dev pytest httpx faker
```

### Simple FastAPI Scaffold

```python
# app/main.py - Tier 1: Simple development scaffold
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging

# Simple in-memory storage for development
fake_db = {}

app = FastAPI(
    title="Governance Platform",
    version="0.1.0-dev",
    docs_url="/docs",  # Swagger enabled for development
)

# Basic models without encryption or complex validation
class Patient(BaseModel):
    id: Optional[str] = None
    name: str
    date_of_birth: str
    diagnosis: str
    
class User(BaseModel):
    username: str
    role: str = "viewer"

# Simple auth for development (no real security)
def get_current_user():
    return User(username="dev_user", role="admin")

@app.get("/health")
def health_check():
    return {"status": "healthy", "tier": "development"}

@app.post("/patients")
def create_patient(patient: Patient):
    # Simple business logic without security
    patient_id = f"PAT-{len(fake_db) + 1:05d}"
    patient.id = patient_id
    fake_db[patient_id] = patient.dict()
    
    logging.info(f"Created patient {patient_id}")
    return {"patient_id": patient_id}

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    if patient_id not in fake_db:
        raise HTTPException(404, "Patient not found")
    return fake_db[patient_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### Configuration Management

```python
# app/config.py - Environment-based configuration
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Tier system
    deployment_tier: str = "development"
    
    # Feature flags
    enable_authentication: bool = False
    enable_encryption: bool = False
    enable_audit_logging: bool = False
    enable_rate_limiting: bool = False
    enable_monitoring: bool = False
    enable_hipaa_compliance: bool = False
    
    # Basic settings
    app_name: str = "Governance Platform"
    debug: bool = True
    
    # Database (optional in Tier 1)
    database_url: Optional[str] = "sqlite:///./test.db"
    
    # External services (mocked in Tier 1)
    redis_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "APP_"

@lru_cache()
def get_settings():
    return Settings()
```

### Development Docker Compose

```yaml
# docker-compose.dev.yml - Tier 1
version: '3.8'

services:
  app:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - APP_DEPLOYMENT_TIER=development
      - APP_DEBUG=true
    command: uvicorn app.main:app --reload --host 0.0.0.0
```

### Simple Dockerfile for Development

```dockerfile
# Dockerfile.dev - Tier 1
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy deps
COPY pyproject.toml .
RUN uv pip install --system -e .

# Dev tools
RUN uv pip install --system ipython ipdb

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0"]
```

## Tier 2: Integration Testing

### Goals
- Add real database connections
- Implement basic authentication
- Enable team collaboration
- Add basic error handling and logging

### Enhanced Dependencies

```bash
# Add Tier 2 dependencies
uv add "sqlalchemy[asyncio]" asyncpg alembic
uv add python-jose[cryptography] passlib[bcrypt]
uv add python-multipart  # For OAuth2 form
uv add structlog
```

### Database Integration

```python
# app/database.py - Tier 2: Real database
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# Use PostgreSQL for integration, SQLite for local
if settings.deployment_tier == "integration":
    SQLALCHEMY_DATABASE_URL = settings.database_url or "postgresql://user:pass@localhost/testdb"
else:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Basic Authentication

```python
# app/auth.py - Tier 2: Simple JWT auth
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import get_settings

settings = get_settings()

# Only enforce in Tier 2+
if settings.enable_authentication:
    SECRET_KEY = "dev-secret-key-change-in-production"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    def create_access_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        return {"username": username, "role": payload.get("role", "viewer")}
else:
    # Mock auth for Tier 1
    async def get_current_user():
        return {"username": "dev_user", "role": "admin"}
```

### Conditional Features with Decorators

```python
# app/features.py - Progressive feature enablement
from functools import wraps
from app.config import get_settings

settings = get_settings()

def requires_tier(min_tier: int):
    """Decorator to conditionally enable features based on tier"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tier_levels = {
                "development": 1,
                "integration": 2,
                "staging": 3,
                "production": 4
            }
            
            current_tier = tier_levels.get(settings.deployment_tier, 1)
            
            if current_tier >= min_tier:
                return await func(*args, **kwargs)
            else:
                # Return mock/simplified version
                return {"message": f"Feature requires tier {min_tier}+"}
        return wrapper
    return decorator

def with_audit(action: str):
    """Conditionally add audit logging"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            if settings.enable_audit_logging:
                # Real audit logging
                await audit_logger.log(action, kwargs)
            else:
                # Just print in development
                print(f"[AUDIT-DEV] {action}: {kwargs}")
            
            return result
        return wrapper
    return decorator
```

### Docker Compose for Integration

```yaml
# docker-compose.integration.yml - Tier 2
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
      POSTGRES_DB: integration_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  app:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      - APP_DEPLOYMENT_TIER=integration
      - APP_DATABASE_URL=postgresql://testuser:testpass@postgres/integration_db
      - APP_REDIS_URL=redis://redis:6379
      - APP_ENABLE_AUTHENTICATION=true
      - APP_ENABLE_AUDIT_LOGGING=true
    ports:
      - "8000:8000"
    command: >
      sh -c "
        alembic upgrade head &&
        uvicorn app.main:app --host 0.0.0.0 --port 8000
      "

volumes:
  postgres_data:
```

## Tier 3: Staging/Dev Deployment

### Goals
- Add security layers (but not full HIPAA)
- Implement monitoring and observability
- Add rate limiting and caching
- Enable performance optimizations

### Additional Dependencies

```bash
# Add Tier 3 dependencies
uv add redis aiocache slowapi
uv add prometheus-fastapi-instrumentator
uv add opentelemetry-api opentelemetry-sdk
uv add sentry-sdk[fastapi]
```

### Security Middleware

```python
# app/middleware.py - Tier 3: Security layers
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import get_settings
import time
import uuid

settings = get_settings()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers in Tier 3+"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        if settings.deployment_tier in ["staging", "production"]:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["X-Request-ID"] = str(uuid.uuid4())
            
            if settings.deployment_tier == "production":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting for Tier 3+"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next):
        if not settings.enable_rate_limiting:
            return await call_next(request)
        
        client_ip = request.client.host
        now = time.time()
        
        # Simple rate limiting (use Redis in production)
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        # Clean old entries
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip]
            if now - timestamp < self.period
        ]
        
        if len(self.clients[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        self.clients[client_ip].append(now)
        return await call_next(request)
```

### Caching Layer

```python
# app/cache.py - Tier 3: Caching implementation
from typing import Optional, Any
import json
import hashlib
from app.config import get_settings

settings = get_settings()

class CacheManager:
    def __init__(self):
        self.backend = self._get_backend()
    
    def _get_backend(self):
        if settings.deployment_tier == "production":
            import redis
            return redis.Redis.from_url(settings.redis_url)
        elif settings.deployment_tier == "staging":
            import redis
            return redis.Redis(host='localhost', decode_responses=True)
        else:
            # In-memory cache for dev/integration
            return {}
    
    async def get(self, key: str) -> Optional[Any]:
        if isinstance(self.backend, dict):
            return self.backend.get(key)
        else:
            value = self.backend.get(key)
            return json.loads(value) if value else None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        if isinstance(self.backend, dict):
            self.backend[key] = value
        else:
            self.backend.setex(key, ttl, json.dumps(value))
    
    def make_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        
        if settings.deployment_tier == "production":
            # Hash keys in production for security
            return hashlib.sha256(key_string.encode()).hexdigest()
        return key_string

cache_manager = CacheManager()
```

### Monitoring Setup

```python
# app/monitoring.py - Tier 3: Observability
from prometheus_fastapi_instrumentator import Instrumentator
from app.config import get_settings
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

settings = get_settings()

def setup_monitoring(app):
    """Progressive monitoring based on tier"""
    
    if settings.deployment_tier == "development":
        # Just basic logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        return
    
    if settings.deployment_tier >= "staging":
        # Prometheus metrics
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_group_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=[".*health.*", ".*metrics.*"],
        )
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    if settings.deployment_tier == "production":
        # Sentry for production
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.deployment_tier,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
            ],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            before_send=filter_sensitive_data,
        )

def filter_sensitive_data(event, hint):
    """Remove PHI from Sentry events"""
    # Filter out sensitive fields
    if 'request' in event and 'data' in event['request']:
        sensitive_fields = ['ssn', 'patient_id', 'diagnosis', 'dob']
        for field in sensitive_fields:
            if field in event['request']['data']:
                event['request']['data'][field] = '[REDACTED]'
    return event
```

### Deployment Configuration

```yaml
# k8s/staging-deployment.yaml - Tier 3
apiVersion: apps/v1
kind: Deployment
metadata:
  name: governance-api-staging
  namespace: staging
spec:
  replicas: 2  # Less replicas than production
  selector:
    matchLabels:
      app: governance-api
      tier: staging
  template:
    metadata:
      labels:
        app: governance-api
        tier: staging
    spec:
      containers:
      - name: api
        image: governance-api:staging
        ports:
        - containerPort: 8000
        env:
        - name: APP_DEPLOYMENT_TIER
          value: "staging"
        - name: APP_ENABLE_AUTHENTICATION
          value: "true"
        - name: APP_ENABLE_RATE_LIMITING
          value: "true"
        - name: APP_ENABLE_MONITORING
          value: "true"
        - name: APP_ENABLE_AUDIT_LOGGING
          value: "true"
        # Not yet enabled
        - name: APP_ENABLE_ENCRYPTION
          value: "false"
        - name: APP_ENABLE_HIPAA_COMPLIANCE
          value: "false"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Tier 4: Production (Full HIPAA Compliance)

### Goals
- Full HIPAA compliance with all security controls
- End-to-end encryption
- Complete audit logging
- High availability and performance
- Disaster recovery

### Production Dependencies

```bash
# Add production dependencies
uv add cryptography hvac  # For encryption and Vault
uv add python-json-logger
uv add circuitbreaker tenacity
uv add ddtrace datadog  # APM
```

### Full Security Implementation

```python
# app/security/hipaa.py - Tier 4: Full HIPAA compliance
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from typing import Dict, Any
import base64
import os
import hvac

class HIPAACompliantEncryption:
    """Production-grade encryption for PHI"""
    
    def __init__(self):
        self.vault_client = hvac.Client(
            url=os.getenv('VAULT_URL'),
            token=os.getenv('VAULT_TOKEN')
        )
        self.backend = default_backend()
    
    def encrypt_phi_field(self, value: str, field_type: str) -> str:
        """AES-256-GCM encryption for PHI fields"""
        # Get field-specific key from Vault
        key_response = self.vault_client.secrets.transit.generate_data_key(
            name='patient-data',
            key_type='aes256-gcm96'
        )
        
        key = base64.b64decode(key_response['data']['plaintext'])
        nonce = os.urandom(12)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(value.encode()) + encryptor.finalize()
        
        # Return base64 encoded: nonce + ciphertext + tag
        encrypted = base64.b64encode(
            nonce + ciphertext + encryptor.tag
        ).decode('ascii')
        
        # Store encrypted key reference
        return f"vault:v1:{key_response['data']['ciphertext']}:{encrypted}"

class HIPAAAuditLogger:
    """Immutable audit logging for HIPAA compliance"""
    
    def __init__(self):
        self.setup_immutable_logging()
    
    async def log_phi_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        resource: str,
        outcome: str,
        request_id: str
    ):
        """Log all PHI access with required HIPAA fields"""
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "patient_id_hash": hashlib.sha256(patient_id.encode()).hexdigest(),
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "request_id": request_id,
            "deployment_tier": "production",
            "compliance_framework": "HIPAA",
            "event_type": "PHI_ACCESS"
        }
        
        # Calculate checksum for tamper detection
        event_string = json.dumps(audit_event, sort_keys=True)
        audit_event["checksum"] = hashlib.sha256(event_string.encode()).hexdigest()
        
        # Send to immutable storage
        await self.send_to_immutable_storage(audit_event)
```

### Production Application Factory

```python
# app/factory.py - Application factory with tier-based configuration
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import get_settings
from app.middleware import SecurityHeadersMiddleware, RateLimitMiddleware
from app.monitoring import setup_monitoring

def create_app() -> FastAPI:
    """Create FastAPI app with tier-appropriate configuration"""
    settings = get_settings()
    
    # Lifespan management based on tier
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if settings.deployment_tier == "production":
            await setup_production_resources(app)
        elif settings.deployment_tier == "staging":
            await setup_staging_resources(app)
        elif settings.deployment_tier == "integration":
            await setup_integration_resources(app)
        else:
            await setup_development_resources(app)
        
        yield
        
        # Shutdown
        await cleanup_resources(app)
    
    # Create app with tier-appropriate settings
    app = FastAPI(
        title=settings.app_name,
        version=get_version(),
        docs_url="/docs" if settings.deployment_tier != "production" else None,
        redoc_url="/redoc" if settings.deployment_tier != "production" else None,
        openapi_url="/openapi.json" if settings.deployment_tier != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware based on tier
    if settings.deployment_tier >= "staging":
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(RateLimitMiddleware)
    
    if settings.deployment_tier == "production":
        from app.security.hipaa import HIPAAMiddleware
        app.add_middleware(HIPAAMiddleware)
    
    # Setup monitoring
    setup_monitoring(app)
    
    # Register routers with tier-based filtering
    register_routers(app, settings.deployment_tier)
    
    return app

async def setup_production_resources(app: FastAPI):
    """Initialize production resources"""
    from app.security.hipaa import HIPAACompliantEncryption, HIPAAAuditLogger
    
    # Initialize encryption
    app.state.encryption = HIPAACompliantEncryption()
    
    # Initialize audit logging
    app.state.audit_logger = HIPAAAuditLogger()
    
    # Setup database with connection pooling
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    app.state.db_engine = create_async_engine(
        settings.database_url,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    app.state.async_session = sessionmaker(
        app.state.db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Initialize Redis with encryption
    import redis
    app.state.redis = redis.Redis(
        connection_pool=redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            max_connections=100,
            decode_responses=False,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL  
                3: 5,  # TCP_KEEPCNT
            }
        )
    )
```

## CI/CD Pipeline with Tier Gates

### GitLab CI Configuration

```yaml
# .gitlab-ci.yml - Progressive CI/CD pipeline
stages:
  - test
  - security-scan
  - build
  - integration-test
  - staging-deploy
  - compliance-check
  - production-deploy

variables:
  DOCKER_DRIVER: overlay2
  UV_CACHE_DIR: $CI_PROJECT_DIR/.cache/uv

# Tier 1: Basic tests (always run)
unit-tests:
  stage: test
  image: python:3.12
  cache:
    paths:
      - .cache/uv
  script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - source $HOME/.cargo/env
    - uv pip install --system -e .[dev]
    - pytest tests/unit -v --cov=app --cov-report=html
  artifacts:
    reports:
      junit: test-results.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

# Tier 2: Integration tests (PR merges)
integration-tests:
  stage: integration-test
  services:
    - postgres:15
    - redis:7
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
    APP_DEPLOYMENT_TIER: integration
    APP_DATABASE_URL: postgresql://test_user:test_pass@postgres/test_db
  script:
    - uv pip install --system -e .[test]
    - alembic upgrade head
    - pytest tests/integration -v
  only:
    - main
    - develop

# Tier 3: Security scanning (staging branches)
security-scan:
  stage: security-scan
  image: aquasec/trivy:latest
  script:
    - trivy fs --severity HIGH,CRITICAL --exit-code 0 .
    - trivy config --severity HIGH,CRITICAL --exit-code 0 .
  artifacts:
    reports:
      security: trivy-results.json
  only:
    - staging
    - main

dependency-check:
  stage: security-scan
  script:
    - pip install safety bandit
    - safety check --json
    - bandit -r app/ -f json -o bandit-report.json
  only:
    - staging
    - main

# Tier 3: Staging deployment
deploy-staging:
  stage: staging-deploy
  environment:
    name: staging
    url: https://staging.governance.example.com
  script:
    - export APP_DEPLOYMENT_TIER=staging
    - export APP_ENABLE_AUTHENTICATION=true
    - export APP_ENABLE_MONITORING=true
    - export APP_ENABLE_RATE_LIMITING=true
    - docker build -t governance-api:staging .
    - docker push $CI_REGISTRY_IMAGE:staging
    - kubectl set image deployment/governance-api governance-api=$CI_REGISTRY_IMAGE:staging -n staging
  only:
    - staging

# Tier 4: HIPAA compliance checks (production only)
hipaa-compliance:
  stage: compliance-check
  script:
    - |
      # Check for required security headers
      curl -I https://staging.governance.example.com | grep -E "Strict-Transport-Security|X-Content-Type-Options|X-Frame-Options"
      
      # Verify encryption is enabled
      python scripts/verify_encryption.py
      
      # Check audit logging
      python scripts/verify_audit_logging.py
      
      # Validate BAA compliance
      python scripts/validate_baa.py
      
      # Security assessment
      python scripts/security_assessment.py
  only:
    - main
  when: manual

# Tier 4: Production deployment (manual approval required)
deploy-production:
  stage: production-deploy
  environment:
    name: production
    url: https://governance.example.com
  script:
    - |
      # Verify all compliance checks passed
      if [ ! -f compliance-approved.flag ]; then
        echo "Compliance checks not approved"
        exit 1
      fi
      
      export APP_DEPLOYMENT_TIER=production
      export APP_ENABLE_HIPAA_COMPLIANCE=true
      export APP_ENABLE_ENCRYPTION=true
      export APP_ENABLE_FULL_AUDIT=true
      
      # Build and deploy
      docker build -t governance-api:$CI_COMMIT_SHA .
      docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
      
      # Blue-green deployment
      kubectl apply -f k8s/production/
      kubectl set image deployment/governance-api-green governance-api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n production
      
      # Health checks
      ./scripts/production_health_check.sh
      
      # Switch traffic
      kubectl patch service governance-api -p '{"spec":{"selector":{"version":"green"}}}' -n production
  only:
    - main
  when: manual
  needs:
    - hipaa-compliance
```

### Environment Configuration Files

```bash
# .env.development (Tier 1)
APP_DEPLOYMENT_TIER=development
APP_DEBUG=true
APP_ENABLE_AUTHENTICATION=false
APP_ENABLE_ENCRYPTION=false
APP_DATABASE_URL=sqlite:///./dev.db

# .env.integration (Tier 2)
APP_DEPLOYMENT_TIER=integration
APP_DEBUG=true
APP_ENABLE_AUTHENTICATION=true
APP_ENABLE_AUDIT_LOGGING=true
APP_DATABASE_URL=postgresql://user:pass@localhost/integration_db
APP_REDIS_URL=redis://localhost:6379

# .env.staging (Tier 3)
APP_DEPLOYMENT_TIER=staging
APP_DEBUG=false
APP_ENABLE_AUTHENTICATION=true
APP_ENABLE_AUDIT_LOGGING=true
APP_ENABLE_RATE_LIMITING=true
APP_ENABLE_MONITORING=true
APP_DATABASE_URL=postgresql://user:pass@pgbouncer:6432/staging_db
APP_REDIS_URL=redis://redis-cluster:6379
APP_SENTRY_DSN=https://xxx@sentry.io/xxx

# .env.production (Tier 4)
APP_DEPLOYMENT_TIER=production
APP_DEBUG=false
APP_ENABLE_AUTHENTICATION=true
APP_ENABLE_ENCRYPTION=true
APP_ENABLE_AUDIT_LOGGING=true
APP_ENABLE_RATE_LIMITING=true
APP_ENABLE_MONITORING=true
APP_ENABLE_HIPAA_COMPLIANCE=true
APP_DATABASE_URL=postgresql://user:pass@pgbouncer:6432/production_db
APP_REDIS_URL=redis://redis-cluster:6379
APP_VAULT_URL=https://vault.internal:8200
APP_SENTRY_DSN=https://xxx@sentry.io/xxx
```

## Migration Scripts Between Tiers

### Tier Migration Validator

```python
# scripts/validate_tier_migration.py
"""
Validates that all required features are enabled before tier migration
"""

import os
import sys
from typing import Dict, List

TIER_REQUIREMENTS = {
    "development": [],
    "integration": [
        "DATABASE_URL",
        "ENABLE_AUTHENTICATION",
    ],
    "staging": [
        "DATABASE_URL",
        "REDIS_URL",
        "ENABLE_AUTHENTICATION",
        "ENABLE_AUDIT_LOGGING",
        "ENABLE_RATE_LIMITING",
        "ENABLE_MONITORING",
    ],
    "production": [
        "DATABASE_URL",
        "REDIS_URL",
        "VAULT_URL",
        "ENABLE_AUTHENTICATION",
        "ENABLE_ENCRYPTION",
        "ENABLE_AUDIT_LOGGING",
        "ENABLE_RATE_LIMITING",
        "ENABLE_MONITORING",
        "ENABLE_HIPAA_COMPLIANCE",
        "SENTRY_DSN",
    ]
}

def validate_tier_migration(current_tier: str, target_tier: str) -> bool:
    """Validate tier migration requirements"""
    
    required_vars = TIER_REQUIREMENTS.get(target_tier, [])
    missing_vars = []
    
    for var in required_vars:
        env_var = f"APP_{var}"
        if not os.getenv(env_var):
            missing_vars.append(env_var)
    
    if missing_vars:
        print(f"❌ Cannot migrate from {current_tier} to {target_tier}")
        print(f"Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    print(f"✅ Ready to migrate from {current_tier} to {target_tier}")
    return True

if __name__ == "__main__":
    current = os.getenv("APP_DEPLOYMENT_TIER", "development")
    target = sys.argv[1] if len(sys.argv) > 1 else "production"
    
    if not validate_tier_migration(current, target):
        sys.exit(1)
```

### Feature Toggle Management

```python
# app/feature_flags.py
"""
Centralized feature flag management across tiers
"""

from enum import Enum
from typing import Dict, Any
from app.config import get_settings

class Feature(Enum):
    AUTHENTICATION = "authentication"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    RATE_LIMITING = "rate_limiting"
    MONITORING = "monitoring"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    WEBSOCKET_SUPPORT = "websocket_support"
    AI_INFERENCE = "ai_inference"
    ADVANCED_SEARCH = "advanced_search"
    BATCH_PROCESSING = "batch_processing"

class FeatureFlags:
    """Manage feature availability across deployment tiers"""
    
    # Define which features are available at each tier
    TIER_FEATURES = {
        "development": {
            Feature.WEBSOCKET_SUPPORT,
            Feature.AI_INFERENCE,
        },
        "integration": {
            Feature.AUTHENTICATION,
            Feature.AUDIT_LOGGING,
            Feature.WEBSOCKET_SUPPORT,
            Feature.AI_INFERENCE,
            Feature.ADVANCED_SEARCH,
        },
        "staging": {
            Feature.AUTHENTICATION,
            Feature.AUDIT_LOGGING,
            Feature.RATE_LIMITING,
            Feature.MONITORING,
            Feature.WEBSOCKET_SUPPORT,
            Feature.AI_INFERENCE,
            Feature.ADVANCED_SEARCH,
            Feature.BATCH_PROCESSING,
        },
        "production": {
            Feature.AUTHENTICATION,
            Feature.ENCRYPTION,
            Feature.AUDIT_LOGGING,
            Feature.RATE_LIMITING,
            Feature.MONITORING,
            Feature.HIPAA_COMPLIANCE,
            Feature.WEBSOCKET_SUPPORT,
            Feature.AI_INFERENCE,
            Feature.ADVANCED_SEARCH,
            Feature.BATCH_PROCESSING,
        }
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.tier = self.settings.deployment_tier
        self.enabled_features = self.TIER_FEATURES.get(self.tier, set())
    
    def is_enabled(self, feature: Feature) -> bool:
        """Check if a feature is enabled for current tier"""
        return feature in self.enabled_features
    
    def require(self, feature: Feature):
        """Decorator to require a feature for an endpoint"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.is_enabled(feature):
                    raise HTTPException(
                        status_code=501,
                        detail=f"Feature {feature.value} not available in {self.tier} tier"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Global instance
feature_flags = FeatureFlags()

# Usage in endpoints
@app.post("/api/ai/inference")
@feature_flags.require(Feature.AI_INFERENCE)
async def ai_inference(request: AIRequest):
    # AI inference logic
    pass
```

## Quick Start Guide

### Starting Development (Tier 1)

```bash
# 1. Clone and setup
git clone <repo>
cd governance-platform

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Install dependencies
uv sync

# 4. Run development server
uv run uvicorn app.main:app --reload

# 5. Access Swagger UI
open http://localhost:8000/docs
```

### Moving to Integration (Tier 2)

```bash
# 1. Start services
docker-compose -f docker-compose.integration.yml up -d

# 2. Run migrations
uv run alembic upgrade head

# 3. Run integration tests
uv run pytest tests/integration

# 4. Check tier migration
python scripts/validate_tier_migration.py integration
```

### Deploying to Staging (Tier 3)

```bash
# 1. Build staging image
docker build -t governance-api:staging .

# 2. Deploy to Kubernetes
kubectl apply -f k8s/staging/

# 3. Verify deployment
kubectl get pods -n staging
kubectl logs -f deployment/governance-api -n staging

# 4. Run smoke tests
python scripts/staging_smoke_tests.py
```

### Production Release (Tier 4)

```bash
# 1. Run compliance checks
./scripts/hipaa_compliance_check.sh

# 2. Build production image
docker build -t governance-api:production .

# 3. Deploy with blue-green strategy
kubectl apply -f k8s/production/blue-green/

# 4. Verify health
./scripts/production_health_check.sh

# 5. Switch traffic
kubectl patch service governance-api \
  -p '{"spec":{"selector":{"version":"green"}}}' \
  -n production
```

## Key Benefits of This Approach

1. **Developer Productivity**: Start simple, add complexity only when needed
2. **Cost Optimization**: Don't pay for production infrastructure during development
3. **Risk Mitigation**: Gradually introduce security and compliance features
4. **Team Collaboration**: Different teams can work at different tiers
5. **Compliance Assurance**: Production gates ensure all requirements are met
6. **Rollback Safety**: Can quickly revert to simpler tiers if issues arise
7. **Testing Efficiency**: Tests appropriate to each tier's complexity
8. **Documentation**: Self-documenting through tier configuration

## Summary

This tiered approach allows you to:
- **Tier 1**: Focus purely on business logic with minimal overhead
- **Tier 2**: Add real infrastructure for team testing
- **Tier 3**: Implement security and performance features
- **Tier 4**: Enable full HIPAA compliance and production hardening

Each tier builds upon the previous one, with clear gates and validation ensuring you never deploy insufficiently secured code to production. The feature flag system and environment-based configuration make it easy to manage what's enabled at each level, while the CI/CD pipeline enforces progressively stricter requirements as code moves toward production.