# Pydantic Settings: From .env Chaos to Type-Safe Configuration

A practical guide for developers transitioning from python-dotenv to Pydantic Settings - the minefield navigation manual.

## The Problem with python-dotenv

If you're coming from the dotenv world, you're used to:
```python
from dotenv import load_dotenv
import os

load_dotenv()

# Everything is a string!
DEBUG = os.getenv("DEBUG")  # "true" as string, not boolean
PORT = os.getenv("PORT")  # "8000" as string, not int
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS")  # How do you parse a list?

# The horror of manual parsing
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")  # Hope no one adds spaces!
```

## Enter Pydantic Settings: Type-Safe Configuration

Pydantic Settings automatically:
- Loads from environment variables
- Converts types (str → int, str → bool, str → list)
- Validates values
- Provides defaults
- Generates documentation

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False  # Automatically converts "true"/"false" to boolean
    port: int = 8000  # Automatically converts "8000" to integer
    allowed_hosts: list[str] = ["localhost"]  # Parses JSON array!
```

## Common Pitfalls and Solutions

### 1. 🔴 Boolean Conversion Confusion

**The Pitfall:**
```python
# .env file
DEBUG=True  # Works
DEBUG=true  # Works
DEBUG=1     # Works
DEBUG=yes   # Works
DEBUG=on    # Works

DEBUG=false  # Works
DEBUG=0      # Works
DEBUG=no     # Works
DEBUG=off    # Works

DEBUG=anything_else  # FAILS! Validation error
```

**The Solution:**
```python
from pydantic import Field, field_validator

class Settings(BaseSettings):
    # Option 1: Use standard boolean conversion
    debug: bool = False
    
    # Option 2: Custom validator for specific needs
    @field_validator("debug", mode="before")
    @classmethod
    def validate_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on", "enabled")
        return bool(v)
```

### 2. 🔴 List/Array Parsing

**The Pitfall:**
```python
# .env file - These DON'T work as expected!
CORS_ORIGINS=http://localhost:3000,http://localhost:8000  # ❌ Treated as string
CORS_ORIGINS="localhost:3000,localhost:8000"  # ❌ Still a string
```

**The Solution:**
```python
# .env file - Use JSON format!
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
# or
CORS_ORIGINS='["http://localhost:3000","http://localhost:8000"]'

class Settings(BaseSettings):
    # Pydantic automatically parses JSON
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # Alternative: Custom parsing for comma-separated
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v
```

### 3. 🔴 Environment Variable Prefixes

**The Pitfall:**
```python
# You want APP_DEBUG but Pydantic looks for DEBUG
class Settings(BaseSettings):
    debug: bool = False  # Looks for DEBUG, not APP_DEBUG
```

**The Solution:**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",  # Now looks for APP_DEBUG
        case_sensitive=False,  # APP_DEBUG or app_debug both work
    )
    
    debug: bool = False  # Looks for APP_DEBUG
```

### 4. 🔴 Missing Required Values

**The Pitfall:**
```python
class Settings(BaseSettings):
    database_url: str  # No default = REQUIRED
    
# If DATABASE_URL is missing → ValidationError on startup!
```

**The Solution:**
```python
from typing import Optional

class Settings(BaseSettings):
    # Option 1: Make it optional
    database_url: Optional[str] = None
    
    # Option 2: Provide a default
    database_url: str = "sqlite:///./test.db"
    
    # Option 3: Use Field for complex defaults
    database_url: str = Field(
        default="sqlite:///./test.db",
        description="Database connection string"
    )
```

### 5. 🔴 Nested Configuration

**The Pitfall:**
```python
# How do you handle complex nested config?
# DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, etc.
```

**The Solution:**
```python
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABASE_")
    
    host: str = "localhost"
    port: int = 5432
    name: str = "myapp"
    user: str = "postgres"
    password: str = "postgres"
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class Settings(BaseSettings):
    # Compose settings
    database: DatabaseSettings = DatabaseSettings()
    debug: bool = False
```

### 6. 🔴 Secret Values

**The Pitfall:**
```python
class Settings(BaseSettings):
    api_key: str = "super-secret"  # Shows up in logs!
    
settings = Settings()
print(settings)  # Oops, prints the API key!
```

**The Solution:**
```python
from pydantic import SecretStr

class Settings(BaseSettings):
    api_key: SecretStr  # Won't be printed
    
settings = Settings()
print(settings.api_key)  # Shows: SecretStr('**********')
print(settings.api_key.get_secret_value())  # Use this to get actual value
```

### 7. 🔴 Environment File Priority

**The Pitfall:**
```python
# Which takes precedence?
# .env file: DEBUG=false
# Environment variable: export DEBUG=true
# Default in code: debug: bool = False
```

**The Solution:**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Priority (highest to lowest):
        # 1. Environment variables
        # 2. .env file
        # 3. Defaults in code
    )
    
    debug: bool = False

# Override for testing
class TestSettings(Settings):
    model_config = SettingsConfigDict(
        env_file=".env.test",  # Different file for tests
    )
```

## Real-World Patterns

### FastAPI Integration

```python
# config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
    )
    
    # Application
    name: str = "My API"
    version: str = "0.1.0"
    debug: bool = False
    
    # Database
    database_url: Optional[str] = None
    
    # Security
    secret_key: SecretStr = SecretStr("change-me-in-production")
    
    # Features
    enable_docs: bool = True
    
    @property
    def docs_enabled(self) -> bool:
        """Disable docs in production."""
        return self.debug or self.enable_docs

@lru_cache()  # Cache settings (singleton pattern)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# main.py
from fastapi import FastAPI
from config import settings

app = FastAPI(
    title=settings.name,
    version=settings.version,
    docs_url="/docs" if settings.docs_enabled else None,
)
```

### Multiple Environment Files

```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load different .env files based on environment
    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'development')}",
        env_prefix="APP_",
    )
    
    database_url: str
    redis_url: str
    secret_key: SecretStr

# Usage:
# ENV=development python main.py  # Loads .env.development
# ENV=production python main.py   # Loads .env.production
```

### Validation with Custom Types

```python
from pydantic import Field, field_validator, HttpUrl
from typing import Literal

class Settings(BaseSettings):
    # URL validation
    webhook_url: HttpUrl  # Must be valid URL
    
    # Enum/choices
    environment: Literal["development", "staging", "production"] = "development"
    
    # Range validation
    port: int = Field(default=8000, ge=1024, le=65535)
    
    # Custom validation
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "mysql://", "sqlite://")):
            raise ValueError("Invalid database URL scheme")
        return v
```

### Dynamic Configuration

```python
class Settings(BaseSettings):
    deployment_tier: str = "development"
    
    # Feature flags based on tier
    @property
    def features(self) -> dict:
        tier_features = {
            "development": {
                "auth": False,
                "cache": False,
                "monitoring": False,
            },
            "staging": {
                "auth": True,
                "cache": True,
                "monitoring": False,
            },
            "production": {
                "auth": True,
                "cache": True,
                "monitoring": True,
            }
        }
        return tier_features.get(self.deployment_tier, {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)
```

## Migration Checklist

When migrating from python-dotenv to Pydantic Settings:

1. **Create Settings Class**
   ```python
   # Before
   from dotenv import load_dotenv
   import os
   load_dotenv()
   DEBUG = os.getenv("DEBUG")
   
   # After
   from pydantic_settings import BaseSettings
   class Settings(BaseSettings):
       debug: bool = False
   settings = Settings()
   ```

2. **Update .env Format**
   ```bash
   # Arrays: Use JSON
   CORS_ORIGINS=["http://localhost:3000"]
   
   # Booleans: Use true/false
   DEBUG=true
   
   # Remove quotes from simple values
   PORT=8000  # Not PORT="8000"
   ```

3. **Add Type Hints**
   ```python
   port: int  # Auto-converts to int
   debug: bool  # Auto-converts to bool
   allowed_hosts: list[str]  # Auto-parses JSON array
   ```

4. **Handle Secrets**
   ```python
   from pydantic import SecretStr
   api_key: SecretStr  # Won't leak in logs
   ```

5. **Add Validation**
   ```python
   port: int = Field(ge=1024, le=65535)
   email: EmailStr
   url: HttpUrl
   ```

## Common Errors and Solutions

### ValidationError on Startup
```python
# Error: field required (type=value_error.missing)
# Solution: Add default or make Optional
database_url: Optional[str] = None
```

### JSON Decode Error
```python
# Error: Expecting property name enclosed in double quotes
# Solution: Use proper JSON in .env
CORS_ORIGINS=["http://localhost:3000"]  # Correct
CORS_ORIGINS=['http://localhost:3000']  # Wrong (single quotes)
```

### Type Conversion Error
```python
# Error: value is not a valid integer
# Solution: Check .env value
PORT=8000  # Correct
PORT=eight-thousand  # Wrong
```

## Best Practices

1. **Use Field() for documentation**
   ```python
   database_url: str = Field(
       default="sqlite:///./test.db",
       description="Database connection string",
       example="postgresql://user:pass@localhost/db"
   )
   ```

2. **Cache settings instance**
   ```python
   @lru_cache()
   def get_settings():
       return Settings()
   ```

3. **Separate settings by environment**
   ```python
   class DevelopmentSettings(Settings):
       debug: bool = True
   
   class ProductionSettings(Settings):
       debug: bool = False
   ```

4. **Validate early, fail fast**
   ```python
   # In main.py
   settings = Settings()  # Fails immediately if config is invalid
   ```

5. **Document your .env.example**
   ```bash
   # Boolean: true, false, 1, 0
   APP_DEBUG=false
   
   # JSON array for lists
   APP_CORS_ORIGINS=["http://localhost:3000"]
   
   # No quotes for numbers
   APP_PORT=8000
   ```

## Summary

Pydantic Settings transforms configuration from:
- **Strings everywhere** → **Type-safe values**
- **Manual parsing** → **Automatic conversion**
- **Runtime errors** → **Startup validation**
- **No documentation** → **Self-documenting**
- **Scattered config** → **Centralized settings**

The learning curve is worth it. Your future self (and your team) will thank you for type-safe, validated configuration that fails fast and documents itself.