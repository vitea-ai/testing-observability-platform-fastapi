# AsyncPG: The Fast PostgreSQL Driver with Sharp Edges

A guide to using asyncpg correctly and avoiding the subtle bugs that will haunt your production system.

## What is AsyncPG?

AsyncPG is the **fastest** PostgreSQL driver for Python, period. It's 3-5x faster than psycopg2 and even faster than Go's pgx. But with great speed comes great responsibility - it's a low-level driver that requires understanding its quirks.

### Speed Comparison
```python
# Benchmark: Fetching 1000 rows
psycopg2:        45ms
psycopg3 (async): 25ms  
asyncpg:          8ms  🚀
```

## When to Use AsyncPG vs Psycopg

### Choose AsyncPG When:
- **High-performance requirements** - You need maximum throughput
- **Simple queries** - Direct SQL without complex ORM needs
- **High concurrency** - Thousands of concurrent connections
- **Read-heavy workloads** - Analytics, reporting, APIs
- **You control the SQL** - Not relying on dynamic ORM queries

### Choose Psycopg2/3 When:
- **Complex dynamic queries** - Heavy ORM usage with SQLAlchemy
- **Legacy codebases** - Existing psycopg2 code
- **Named parameters needed** - Queries with many parameters
- **Database agnostic code** - Need to support multiple databases
- **Team familiarity** - Team knows psycopg patterns
- **Advanced PostgreSQL features** - LISTEN/NOTIFY, custom types
- **Simpler mental model** - Dict-like results, forgiving API

### FastAPI-Specific Considerations

**AsyncPG + FastAPI Issues:**
```python
# PROBLEM: SQLAlchemy ORM doesn't work with asyncpg directly
# You need special setup with async SQLAlchemy

# Won't work - mixing sync SQLAlchemy with asyncpg
from sqlalchemy.orm import Session
Session.query(User).filter()  # ❌ Sync ORM with async driver

# Need async SQLAlchemy setup (complex!)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
engine = create_async_engine("postgresql+asyncpg://...")
```

**Psycopg3 + FastAPI (Often Better Choice):**
```python
# Psycopg3 works seamlessly with SQLAlchemy async
from sqlalchemy.ext.asyncio import create_async_engine

# Simple setup - just works
engine = create_async_engine("postgresql+psycopg://...")

# Supports both async and sync patterns
async with AsyncSession(engine) as session:
    result = await session.execute(select(User))
    
# And you get named parameters!
await session.execute(
    text("SELECT * FROM users WHERE name = :name"),
    {"name": "John"}  # ✅ Works!
)
```

## The Fundamental Difference

### Traditional Drivers (psycopg2, SQLAlchemy)
```python
# Everything is Pythonic and forgiving
cursor.execute("SELECT * FROM users WHERE age > %s", (18,))
cursor.execute("SELECT * FROM users WHERE age > %(age)s", {"age": 18})
# Returns: Dict-like objects or tuples
```

### AsyncPG
```python
# Strict, typed, and blazing fast
await conn.fetch("SELECT * FROM users WHERE age > $1", 18)
# Returns: asyncpg.Record objects (not dicts!)
# No named parameters, only positional $1, $2, $3...
```

## Critical Things to Watch Out For

### 1. 🔴 Records Are NOT Dictionaries

**The Problem:**
```python
# This WILL break your code
result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)
user_dict = dict(result)  # ✅ Works
json.dumps(result)        # ❌ FAILS! Record is not JSON serializable

# Also breaks
result['name'] = 'New Name'  # ❌ FAILS! Records are immutable
```

**The Solution:**
```python
# Convert to dict immediately if needed
result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)
user = dict(result) if result else None

# Or use a transformer
def record_to_dict(record):
    return dict(record) if record else None

users = await conn.fetch("SELECT * FROM users")
users_list = [dict(user) for user in users]
```

### 2. 🔴 No Named Parameters

**The Problem:**
```python
# This doesn't work
await conn.fetch(
    "SELECT * FROM users WHERE name = :name AND age > :age",
    name="John", age=18  # ❌ FAILS!
)
```

**The Solution:**
```python
# Use numbered placeholders
await conn.fetch(
    "SELECT * FROM users WHERE name = $1 AND age > $2",
    "John", 18  # ✅ Positional arguments
)

# For many parameters, use a helper
def build_query(filters):
    conditions = []
    values = []
    for i, (key, value) in enumerate(filters.items(), 1):
        conditions.append(f"{key} = ${i}")
        values.append(value)
    
    query = f"SELECT * FROM users WHERE {' AND '.join(conditions)}"
    return query, values

query, values = build_query({"name": "John", "age": 18})
await conn.fetch(query, *values)
```

### 3. 🔴 Connection Pool Gotchas

**The Problem:**
```python
# This leaks connections!
pool = await asyncpg.create_pool(DATABASE_URL)
conn = await pool.acquire()
result = await conn.fetch("SELECT * FROM users")
# Forgot to release! Connection is leaked
```

**The Solution:**
```python
# Always use context manager
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT * FROM users")
    # Connection automatically released

# Or for transactions
async with pool.acquire() as conn:
    async with conn.transaction():
        await conn.execute("INSERT INTO users ...")
        await conn.execute("UPDATE stats ...")
        # Automatically commits or rolls back
```

### 4. 🔴 Type Conversions Are Strict

**The Problem:**
```python
# PostgreSQL JSONB != Python dict automatically
result = await conn.fetchrow("SELECT data FROM configs WHERE id = $1", 1)
config = result['data']  # This is already a dict! (asyncpg converts JSONB)

# But ARRAY types might surprise you
result = await conn.fetchrow("SELECT tags FROM posts WHERE id = $1", 1)
tags = result['tags']  # This is a Python list, not string!
```

**The Solution:**
```python
# Understand automatic conversions:
# - JSONB/JSON → dict
# - ARRAY → list
# - TIMESTAMP → datetime
# - DATE → date
# - NUMERIC → Decimal (not float!)
# - BIGINT → int (Python handles big ints)

# For custom types, set up codecs
await conn.set_type_codec(
    'json',
    encoder=json.dumps,
    decoder=json.loads,
    schema='pg_catalog'
)
```

### 5. 🔴 Prepared Statements Cache

**The Problem:**
```python
# Dynamic queries can exhaust prepared statement cache
for table in tables:
    # Each unique query string creates a new prepared statement
    await conn.fetch(f"SELECT * FROM {table}")  # ❌ Cache pollution
```

**The Solution:**
```python
# Reuse query patterns
query = "SELECT * FROM users WHERE status = $1"
active = await conn.fetch(query, 'active')
inactive = await conn.fetch(query, 'inactive')  # Reuses prepared statement

# For dynamic queries, disable preparation
await conn.fetch(
    f"SELECT * FROM {table}",
    prepare=False  # Don't cache this query
)
```

### 6. 🔴 Transaction Isolation

**The Problem:**
```python
# Default isolation can cause issues
async with conn.transaction():
    user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)
    # Another transaction updates this user
    await asyncio.sleep(1)
    await conn.execute("UPDATE users SET credits = credits + 10 WHERE id = $1", 1)
    # May cause lost update!
```

**The Solution:**
```python
# Specify isolation level
async with conn.transaction(isolation='serializable'):
    # Now protected against concurrent modifications
    user = await conn.fetchrow("SELECT * FROM users WHERE id = $1 FOR UPDATE", 1)
    await conn.execute("UPDATE users SET credits = $1", user['credits'] + 10)
```

### 7. 🔴 FastAPI + SQLAlchemy + AsyncPG Complexity

**The Problem:**
```python
# Setting up SQLAlchemy with asyncpg is complex
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# This setup is tricky to get right
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True,
    pool_pre_ping=True,  # Doesn't work the same way!
    pool_size=20,
)

# Session factory setup is different
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Dependency injection is more complex
async def get_db():
    async with async_session() as session:
        yield session  # Different from sync version
```

**Simpler with Psycopg3:**
```python
# Psycopg3 is more straightforward
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+psycopg://user:pass@localhost/db",
    # All standard options work as expected
    pool_pre_ping=True,
    pool_size=20,
)
# That's it! Works like sync SQLAlchemy but async
```

## Common Patterns

### FastAPI + AsyncPG Setup (Raw SQL)

```python
# database.py
import asyncpg
from contextlib import asynccontextmanager

class Database:
    pool: asyncpg.Pool = None
    
    @classmethod
    async def connect(cls):
        cls.pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=10,
            max_size=20,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60,
        )
    
    @classmethod
    async def disconnect(cls):
        if cls.pool:
            await cls.pool.close()
    
    @classmethod
    @asynccontextmanager
    async def acquire(cls):
        async with cls.pool.acquire() as conn:
            yield conn
    
    @classmethod
    @asynccontextmanager
    async def transaction(cls):
        async with cls.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    await Database.connect()
    yield
    await Database.disconnect()

app = FastAPI(lifespan=lifespan)

# Usage in endpoints
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with Database.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT * FROM users WHERE id = $1", 
            user_id
        )
        return dict(user) if user else None
```

### Bulk Operations

```python
# FAST: Use copy_records_to_table
records = [
    (1, 'Alice', 25),
    (2, 'Bob', 30),
    (3, 'Charlie', 35)
]

async with Database.acquire() as conn:
    await conn.copy_records_to_table(
        'users',
        records=records,
        columns=['id', 'name', 'age']
    )

# Or executemany for updates
await conn.executemany(
    "UPDATE users SET last_seen = $2 WHERE id = $1",
    [(1, datetime.now()), (2, datetime.now())]
)
```

### Query Builder Pattern

```python
class UserQuery:
    @staticmethod
    async def find_by_id(conn: asyncpg.Connection, user_id: int):
        return await conn.fetchrow(
            "SELECT * FROM users WHERE id = $1",
            user_id
        )
    
    @staticmethod
    async def find_active(conn: asyncpg.Connection, limit: int = 10):
        return await conn.fetch(
            """
            SELECT * FROM users 
            WHERE active = true 
            ORDER BY created_at DESC 
            LIMIT $1
            """,
            limit
        )
    
    @staticmethod
    async def update_last_seen(conn: asyncpg.Connection, user_id: int):
        return await conn.execute(
            "UPDATE users SET last_seen = NOW() WHERE id = $1",
            user_id
        )
```

## Performance Tips

### 1. Use Fetch Methods Correctly

```python
# fetchrow - Single row (returns Record or None)
user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)

# fetch - Multiple rows (returns List[Record])
users = await conn.fetch("SELECT * FROM users LIMIT 10")

# fetchval - Single value (returns value or None)
count = await conn.fetchval("SELECT COUNT(*) FROM users")

# execute - No return value (returns status string)
await conn.execute("UPDATE users SET active = true")
```

### 2. Connection Pool Settings

```python
pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=10,          # Minimum connections
    max_size=20,          # Maximum connections
    max_queries=50000,    # Queries before connection reset
    max_inactive_connection_lifetime=300,  # 5 minutes
    command_timeout=60,   # Query timeout
    server_settings={
        'application_name': 'myapp',
        'jit': 'off'  # Disable JIT for consistent performance
    },
)
```

### 3. Use COPY for Bulk Inserts

```python
# SLOW: Individual inserts
for user in users:
    await conn.execute(
        "INSERT INTO users (name, email) VALUES ($1, $2)",
        user['name'], user['email']
    )

# FAST: COPY command
await conn.copy_records_to_table(
    'users',
    records=[(u['name'], u['email']) for u in users],
    columns=['name', 'email']
)
# 100x faster for large datasets!
```

## Error Handling

```python
import asyncpg
from asyncpg import PostgresError, UniqueViolationError

try:
    async with Database.transaction() as conn:
        await conn.execute("INSERT INTO users (email) VALUES ($1)", email)
except UniqueViolationError as e:
    # Handle duplicate email
    raise ValueError(f"Email {email} already exists")
except PostgresError as e:
    # Handle other PostgreSQL errors
    logger.error(f"Database error: {e}")
    raise
except asyncpg.DataError as e:
    # Invalid data type
    raise ValueError(f"Invalid data: {e}")
except asyncpg.IntegrityConstraintViolationError as e:
    # Foreign key, check constraint, etc.
    raise ValueError(f"Constraint violation: {e}")
```

## Testing with AsyncPG

```python
import pytest
import asyncpg
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def mock_db_connection():
    """Mock asyncpg connection for testing"""
    conn = AsyncMock(spec=asyncpg.Connection)
    
    # Mock fetchrow to return dict-like object
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: {"id": 1, "name": "Test"}[key]
    mock_record.keys = lambda self: ["id", "name"]
    conn.fetchrow.return_value = mock_record
    
    return conn

async def test_get_user(mock_db_connection):
    # Your test using mocked connection
    result = await mock_db_connection.fetchrow(
        "SELECT * FROM users WHERE id = $1", 1
    )
    assert result['id'] == 1
```

## Summary: Key Rules for AsyncPG

1. **Records are immutable** - Convert to dict if you need mutability
2. **Use $1, $2, $3** - No named parameters
3. **Always release connections** - Use context managers
4. **Types are strictly converted** - Know what you're getting
5. **Pool configuration matters** - Set appropriate limits
6. **Use COPY for bulk operations** - It's incredibly fast
7. **Handle specific exceptions** - PostgresError subclasses
8. **Prepare statements wisely** - Watch your cache
9. **Transaction isolation matters** - Especially for financial data
10. **Test with proper mocks** - AsyncPG Records are special

## The Verdict

**Use AsyncPG when:**
- You need absolute maximum performance
- You're comfortable with raw SQL
- You have simple, well-defined queries
- You're building read-heavy APIs

**Use Psycopg3 when:**
- You need SQLAlchemy ORM integration
- You want a gentler learning curve
- You have complex dynamic queries
- Your team values maintainability over raw speed

Remember: AsyncPG trades convenience for speed. If you need ORM features or a forgiving API, use Psycopg3 with async SQLAlchemy. If you need raw speed and can handle the sharp edges, AsyncPG is unbeatable.