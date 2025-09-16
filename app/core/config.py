"""
Core configuration management with tier-based feature flags.

This module provides centralized configuration for the FastAPI application
with progressive enhancement through deployment tiers.
"""

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploymentTier(str, Enum):
    """Deployment tier levels."""
    DEVELOPMENT = "development"
    INTEGRATION = "integration"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with tier-based configuration.
    
    Settings are loaded from environment variables with the APP_ prefix.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================
    # Core Application Settings
    # ==========================================
    name: str = Field(default="Testing Observability Platform", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    env: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=True, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8000, description="Port to bind")
    
    # Deployment tier - controls feature availability
    deployment_tier: DeploymentTier = Field(
        default=DeploymentTier.DEVELOPMENT,
        description="Deployment tier level"
    )
    
    # ==========================================
    # Feature Flags (Progressive Enhancement)
    # ==========================================
    enable_authentication: bool = Field(default=False)
    enable_encryption: bool = Field(default=False)
    enable_audit_logging: bool = Field(default=False)
    enable_rate_limiting: bool = Field(default=False)
    enable_monitoring: bool = Field(default=False)
    enable_hipaa_compliance: bool = Field(default=False)
    enable_websocket: bool = Field(default=False)
    enable_background_tasks: bool = Field(default=False)
    enable_caching: bool = Field(default=False)
    
    # ==========================================
    # API Configuration
    # ==========================================
    api_v1_prefix: str = Field(default="/api/v1")
    allowed_hosts: List[str] = Field(default=["*"])
    cors_origins: List[str] = Field(
        default=["*"]
    )
    
    @field_validator('allowed_hosts', 'cors_origins', mode='before')
    @classmethod
    def parse_json_list(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse JSON strings into lists for environment variables."""
        if isinstance(v, str):
            try:
                # Try to parse as JSON array
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]  # Single value becomes a list
            except json.JSONDecodeError:
                # If not JSON, treat as comma-separated
                if ',' in v:
                    return [item.strip() for item in v.split(',')]
                # Single value
                return [v.strip()]
        return v
    
    # ==========================================
    # Database Configuration (Tier 2+)
    # ==========================================
    database_url: Optional[str] = Field(
        default="postgresql://dbadmin:postgres123@localhost:5433/evaluation_db",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20)
    database_max_overflow: int = Field(default=30)
    database_echo: bool = Field(default=False)
    
    # ==========================================
    # Redis Configuration (Tier 3+)
    # ==========================================
    redis_url: Optional[str] = Field(default=None)
    redis_max_connections: int = Field(default=100)
    
    # ==========================================
    # Celery Configuration (for async task processing)
    # ==========================================
    celery_broker_url: str = Field(
        default="redis://:vitea_redis_dev_2024@localhost:6379/3",
        description="Celery broker URL (Redis)"
    )
    celery_result_backend: str = Field(
        default="redis://:vitea_redis_dev_2024@localhost:6379/4",
        description="Celery result backend URL"
    )
    celery_task_default_queue: str = Field(
        default="evaluations",
        description="Default queue for Celery tasks"
    )
    celery_worker_concurrency: int = Field(
        default=4,
        description="Number of concurrent Celery workers"
    )
    celery_task_time_limit: int = Field(
        default=600,
        description="Maximum time for task execution in seconds"
    )
    
    # ==========================================
    # Security Configuration
    # ==========================================
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT encoding"
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    
    # Rate limiting (Tier 3+)
    rate_limit_calls: int = Field(default=100)
    rate_limit_period: int = Field(default=60)
    
    # ==========================================
    # Monitoring Configuration (Tier 3+)
    # ==========================================
    sentry_dsn: Optional[str] = Field(default=None)
    prometheus_enabled: bool = Field(default=False)
    opentelemetry_enabled: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")
    
    # ==========================================
    # Encryption Configuration (Tier 4)
    # ==========================================
    vault_url: Optional[str] = Field(default=None)
    vault_token: Optional[str] = Field(default=None)
    enable_field_encryption: bool = Field(default=False)
    
    # ==========================================
    # Testing Configuration
    # ==========================================
    testing: bool = Field(default=False)
    test_database_url: str = Field(default="sqlite:///./test.db")
    
    # ==========================================
    # Platform-Specific Configuration
    # ==========================================
    # Evaluator Service
    evaluator_service_url: str = Field(
        default="http://localhost:9002",
        description="URL for the AI evaluator service"
    )
    evaluator_timeout: int = Field(default=300, description="Timeout for evaluator calls in seconds")
    
    # Experiment Configuration
    max_concurrent_evaluations: int = Field(default=5)
    default_batch_size: int = Field(default=10)
    experiment_timeout: int = Field(default=3600, description="Max experiment runtime in seconds")
    
    # Healthcare Compliance
    enable_healthcare_mode: bool = Field(default=True)
    phi_detection_enabled: bool = Field(default=True)
    audit_retention_days: int = Field(default=2555, description="7 years for HIPAA")
    
    @field_validator("deployment_tier", mode="before")
    def validate_deployment_tier(cls, v: str) -> str:
        """Validate and convert deployment tier string."""
        if isinstance(v, str):
            v = v.lower()
        if v not in [tier.value for tier in DeploymentTier]:
            raise ValueError(f"Invalid deployment tier: {v}")
        return v
    
    def get_enabled_features(self) -> Dict[str, bool]:
        """Get all enabled features based on tier and configuration."""
        tier_features = self._get_tier_features()
        
        # Override with explicit settings
        features = {
            "authentication": self.enable_authentication,
            "encryption": self.enable_encryption,
            "audit_logging": self.enable_audit_logging,
            "rate_limiting": self.enable_rate_limiting,
            "monitoring": self.enable_monitoring,
            "hipaa_compliance": self.enable_hipaa_compliance,
            "websocket": self.enable_websocket,
            "background_tasks": self.enable_background_tasks,
            "caching": self.enable_caching,
        }
        
        # Apply tier minimums
        for feature, enabled in tier_features.items():
            if enabled:
                features[feature] = True
                
        return features
    
    def _get_tier_features(self) -> Dict[str, bool]:
        """Get minimum features enabled for current tier."""
        tier_configs = {
            DeploymentTier.DEVELOPMENT: {
                "authentication": False,
                "encryption": False,
                "audit_logging": False,
                "rate_limiting": False,
                "monitoring": False,
                "hipaa_compliance": False,
            },
            DeploymentTier.INTEGRATION: {
                "authentication": False,  # Disabled for testing
                "encryption": False,
                "audit_logging": True,
                "rate_limiting": False,
                "monitoring": False,
                "hipaa_compliance": False,
            },
            DeploymentTier.STAGING: {
                "authentication": True,
                "encryption": False,
                "audit_logging": True,
                "rate_limiting": True,
                "monitoring": True,
                "hipaa_compliance": False,
            },
            DeploymentTier.PRODUCTION: {
                "authentication": True,
                "encryption": True,
                "audit_logging": True,
                "rate_limiting": True,
                "monitoring": True,
                "hipaa_compliance": True,
            },
        }
        
        return tier_configs.get(self.deployment_tier, {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        features = self.get_enabled_features()
        return features.get(feature, False)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.deployment_tier == DeploymentTier.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.deployment_tier == DeploymentTier.PRODUCTION
    
    @property
    def docs_enabled(self) -> bool:
        """Check if API documentation should be enabled."""
        return self.deployment_tier != DeploymentTier.PRODUCTION
    
    def get_database_url(self) -> str:
        """Get the appropriate database URL based on environment."""
        if self.testing:
            return self.test_database_url
        return self.database_url or "sqlite:///./test.db"
    
    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL with fallback to redis_url."""
        if self.redis_url:
            # Use redis_url if provided (for production environments)
            return self.redis_url.replace("/2", "/3")  # Use database 3 for Celery
        return self.celery_broker_url

    def get_celery_result_backend(self) -> str:
        """Get Celery result backend URL with fallback to redis_url."""
        if self.redis_url:
            # Use redis_url if provided (for production environments)
            return self.redis_url.replace("/2", "/4")  # Use database 4 for results
        return self.celery_result_backend


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Export settings instance for easy import
settings = get_settings()
