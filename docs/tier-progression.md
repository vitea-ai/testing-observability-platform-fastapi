# FastAPI Tiered Architecture Guide

## Overview

This FastAPI template implements a **4-tier progressive enhancement strategy** that allows you to start simple and add complexity as needed. Each tier builds upon the previous one, enabling smooth transitions from development to production.

## Tier System

### 🚀 Tier 1: Local Development
**Goal:** Rapid prototyping with minimal setup

**Features:**
- Simple FastAPI application
- In-memory data storage
- No authentication required
- Swagger UI enabled
- Hot reload enabled
- Mock external services

**When to use:**
- Initial development
- Proof of concepts
- Learning FastAPI
- Quick demos

**Configuration:**
```bash
APP_DEPLOYMENT_TIER=development
APP_ENABLE_AUTHENTICATION=false
APP_DATABASE_URL=sqlite:///./test.db
```

### 🔧 Tier 2: Integration Testing  
**Goal:** Team collaboration with real infrastructure

**Features:**
- PostgreSQL database
- Basic JWT authentication
- Database migrations with Alembic
- Audit logging
- Integration tests
- Docker Compose setup

**When to use:**
- Team development
- Integration testing
- Feature development
- API testing

**Configuration:**
```bash
APP_DEPLOYMENT_TIER=integration
APP_ENABLE_AUTHENTICATION=true
APP_ENABLE_AUDIT_LOGGING=true
APP_DATABASE_URL=postgresql://user:pass@localhost/db
```

### 📊 Tier 3: Staging Deployment
**Goal:** Production-like environment with monitoring

**Features:**
- Redis caching
- Rate limiting
- Security headers
- Prometheus metrics
- Sentry error tracking
- Performance optimizations
- Kubernetes-ready

**When to use:**
- UAT testing
- Performance testing
- Security testing
- Pre-production validation

**Configuration:**
```bash
APP_DEPLOYMENT_TIER=staging
APP_ENABLE_RATE_LIMITING=true
APP_ENABLE_MONITORING=true
APP_REDIS_URL=redis://localhost:6379
APP_SENTRY_DSN=your-sentry-dsn
```

### 🏢 Tier 4: Production
**Goal:** Full production readiness with compliance

**Features:**
- End-to-end encryption
- HIPAA compliance (if needed)
- Full audit logging
- High availability
- Disaster recovery
- Vault integration
- Complete monitoring

**When to use:**
- Production deployment
- Regulated environments
- High-security requirements

**Configuration:**
```bash
APP_DEPLOYMENT_TIER=production
APP_ENABLE_ENCRYPTION=true
APP_ENABLE_HIPAA_COMPLIANCE=true
APP_VAULT_URL=https://vault.internal:8200
```

## Progressive Enhancement Strategy

### Moving from Tier 1 to Tier 2

1. **Add Database:**
   ```bash
   # Start PostgreSQL
   docker run -d \
     --name postgres \
     -e POSTGRES_PASSWORD=password \
     -p 5432:5432 \
     postgres:15
   
   # Update .env
   APP_DEPLOYMENT_TIER=integration
   APP_DATABASE_URL=postgresql://postgres:password@localhost/mydb
   ```

2. **Enable Authentication:**
   ```bash
   APP_ENABLE_AUTHENTICATION=true
   ```

3. **Run Migrations:**
   ```bash
   uv run alembic upgrade head
   ```

### Moving from Tier 2 to Tier 3

1. **Add Redis:**
   ```bash
   docker run -d --name redis -p 6379:6379 redis:7
   APP_REDIS_URL=redis://localhost:6379
   ```

2. **Enable Security Features:**
   ```bash
   APP_ENABLE_RATE_LIMITING=true
   APP_ENABLE_MONITORING=true
   ```

3. **Setup Monitoring:**
   - Configure Prometheus endpoint
   - Add Sentry DSN
   - Enable metrics collection

### Moving from Tier 3 to Tier 4

1. **Security Hardening:**
   - Enable encryption
   - Setup Vault
   - Configure HIPAA compliance

2. **Infrastructure:**
   - Deploy to Kubernetes
   - Setup load balancing
   - Configure auto-scaling

3. **Compliance:**
   - Enable full audit logging
   - Implement data retention policies
   - Setup backup and recovery

## Feature Flags

The template uses feature flags to progressively enable functionality:

| Feature | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------|--------|--------|--------|--------|
| Authentication | ❌ | ✅ | ✅ | ✅ |
| Database | Optional | ✅ | ✅ | ✅ |
| Caching | ❌ | ❌ | ✅ | ✅ |
| Rate Limiting | ❌ | ❌ | ✅ | ✅ |
| Monitoring | ❌ | ❌ | ✅ | ✅ |
| Encryption | ❌ | ❌ | ❌ | ✅ |
| HIPAA Compliance | ❌ | ❌ | ❌ | ✅ |

## Development Workflow

### 1. Initial Setup
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone template
git clone <repository>
cd fastapi-template

# Install dependencies
uv sync

# Copy environment
cp .env.example .env

# Run development server
uv run uvicorn app.main:app --reload
```

### 2. Adding Dependencies
```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add tier-specific dependency
uv add --group staging prometheus-client
```

### 3. Running Tests
```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit

# Integration tests
uv run pytest tests/integration

# With coverage
uv run pytest --cov=app --cov-report=html
```

### 4. Database Migrations
```bash
# Create migration
uv run alembic revision --autogenerate -m "Add user table"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

## Docker Deployment

### Development
```bash
docker-compose -f docker-compose.dev.yml up
```

### Integration
```bash
docker-compose -f docker-compose.integration.yml up
```

### Production Build
```bash
docker build -f docker/Dockerfile.prod -t myapp:latest .
docker run -p 8000:8000 --env-file .env.production myapp:latest
```

## Best Practices

### 1. Environment Variables
- Never commit `.env` files
- Use `.env.example` as template
- Store secrets in vault/secret manager
- Use different values per tier

### 2. Database
- Always use migrations
- Never use `Base.metadata.create_all()` in production
- Use connection pooling
- Implement retry logic

### 3. Security
- Enable security headers in staging/production
- Use rate limiting
- Implement proper authentication
- Audit sensitive operations

### 4. Monitoring
- Add health checks
- Export metrics
- Log structured data
- Track errors with Sentry

### 5. Testing
- Write tests for each tier
- Mock external dependencies
- Test feature flags
- Validate tier transitions

## Troubleshooting

### Common Issues

1. **Database connection failed:**
   - Check DATABASE_URL format
   - Verify database is running
   - Check network connectivity

2. **Redis connection failed:**
   - Verify Redis is running
   - Check REDIS_URL format
   - Ensure Redis is accessible

3. **Authentication not working:**
   - Check ENABLE_AUTHENTICATION flag
   - Verify SECRET_KEY is set
   - Check token expiration

4. **Rate limiting not working:**
   - Enable ENABLE_RATE_LIMITING
   - Check Redis connection
   - Verify middleware order

## Performance Optimization

### Tier-based Optimizations

**Tier 1-2:**
- Use SQLite for development
- Minimal middleware
- Debug mode enabled

**Tier 3:**
- Add Redis caching
- Enable connection pooling
- Use async database operations

**Tier 4:**
- Enable response compression
- Use CDN for static files
- Implement database read replicas
- Use message queues for async tasks

## Security Considerations

### Per-Tier Security

**Tier 1:**
- Basic input validation
- No authentication required

**Tier 2:**
- JWT authentication
- Password hashing
- Basic audit logging

**Tier 3:**
- Rate limiting
- Security headers
- CORS configuration
- Input sanitization

**Tier 4:**
- Field-level encryption
- Vault integration
- Complete audit trail
- Compliance controls

## Conclusion

This tiered approach allows you to:
- Start fast with minimal complexity
- Add features progressively
- Maintain clean separation of concerns
- Deploy with confidence at each tier

Choose the appropriate tier for your current needs and upgrade as requirements evolve.
