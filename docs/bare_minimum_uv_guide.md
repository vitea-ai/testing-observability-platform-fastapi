# The Bare Minimum uv Guide

A practical guide to understanding why `uv` is revolutionizing Python development and how to use it effectively.

## Why uv Over pip and Others?

### Speed Comparison
- **uv**: 10-100x faster than pip
- **pip**: Traditional, but slow
- **poetry**: Feature-rich but sluggish  
- **pipenv**: Slow and problematic
- **pip-tools**: Better than pip, but still slow

### Real Performance Numbers
```bash
# Installing pandas + numpy + matplotlib
pip:        ~15 seconds
poetry:     ~12 seconds
uv:         ~0.8 seconds  🚀
```

### Key Advantages

#### 1. **All-in-One Tool**
- Python version management (like pyenv)
- Virtual environment creation (like venv)
- Dependency management (like pip + pip-tools)
- Lock file generation (like poetry)
- Script running (like npm scripts)

#### 2. **Zero Setup**
```bash
# With pip, you need:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# With uv, just:
uv sync
```

#### 3. **Rust-Powered Performance**
- Written in Rust for maximum speed
- Parallel downloads and installations
- Efficient dependency resolution
- Smart caching system

#### 4. **Reproducible Builds**
- Automatic lock file generation
- Cross-platform compatibility
- Deterministic installations
- Python version pinning

## Process Differences: pip vs uv

### Starting a New Project

**Old Way (pip):**
```bash
# 1. Check Python version
python --version

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install fastapi uvicorn

# 6. Freeze requirements
pip freeze > requirements.txt
```

**New Way (uv):**
```bash
# 1. Create pyproject.toml with dependencies
# 2. Run:
uv sync

# Done! ✨
```

### Installing Dependencies

**pip:**
```bash
# Must activate venv first
source .venv/bin/activate
pip install requests
pip freeze > requirements.txt  # Manual update
```

**uv:**
```bash
# No activation needed
uv add requests  # Automatically updates pyproject.toml
```

### Running Scripts

**pip:**
```bash
source .venv/bin/activate
python script.py
# or
.venv/bin/python script.py
```

**uv:**
```bash
uv run python script.py  # Automatically uses correct environment
```

## Essential uv Commands Everyone Should Know

### 1. Project Setup
```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project (creates venv + installs deps)
uv sync
```

### 2. Dependency Management
```bash
# Add a package
uv add fastapi

# Add dev dependency
uv add --dev pytest

# Remove a package
uv remove fastapi

# Update all packages
uv sync --upgrade
```

### 3. Running Code
```bash
# Run Python script
uv run python app.py

# Run any command in the environment
uv run pytest
uv run black .
uv run mypy src/
```

### 4. Working with Different Python Versions
```bash
# Use specific Python version (downloads if needed!)
uv python pin 3.12  # Sets Python 3.12 for this project
uv sync              # Creates venv with Python 3.12

# List available Python versions
uv python list
```

### 5. Installing from requirements.txt (Migration)
```bash
# If you have an existing requirements.txt
uv pip install -r requirements.txt

# Better: Convert to pyproject.toml
uv add $(cat requirements.txt | tr '\n' ' ')
```

### 6. Lock Files
```bash
# uv automatically creates uv.lock for reproducibility
# Commit this file to git!

# Install exact versions from lock
uv sync --frozen  # Won't update lock file
```

## Quick Migration Guide

### From pip to uv

1. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create pyproject.toml:**
   ```toml
   [project]
   name = "my-project"
   version = "0.1.0"
   requires-python = ">=3.11"
   dependencies = [
       # Add your dependencies here
   ]
   
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   ```

3. **Add your dependencies:**
   ```bash
   # If you have requirements.txt
   uv add $(cat requirements.txt | grep -v '^#' | tr '\n' ' ')
   
   # Or add manually
   uv add fastapi uvicorn sqlalchemy
   ```

4. **Run your app:**
   ```bash
   uv run python main.py
   ```

## Common Scenarios

### Development Workflow
```bash
# Morning setup
git pull
uv sync          # Gets you up to date

# Add new feature requiring a package
uv add httpx
# Code...
git add .
git commit -m "Add httpx integration"
```

### CI/CD Pipeline
```yaml
# GitHub Actions example
- name: Install uv
  uses: astral-sh/setup-uv@v3
  
- name: Install dependencies
  run: uv sync
  
- name: Run tests
  run: uv run pytest
```

### Docker
```dockerfile
# Multi-stage build with uv
FROM ghcr.io/astral-sh/uv:python3.12-slim as builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "main.py"]
```

## Tips and Tricks

1. **Global Tools**: Install tools globally with uvx
   ```bash
   uvx ruff check .  # Run ruff without installing
   ```

2. **Aliases**: Add to your shell config
   ```bash
   alias uvr="uv run"
   alias uvs="uv sync"
   alias uva="uv add"
   ```

3. **Multiple Python Versions**: Test across versions
   ```bash
   uv run --python 3.11 pytest
   uv run --python 3.12 pytest
   ```

4. **Workspace Management**: Monorepo support
   ```toml
   [tool.uv.workspace]
   members = ["packages/*"]
   ```

## Common Gotchas

1. **Don't activate venv manually** - uv handles this
2. **Commit uv.lock** - Essential for reproducible builds
3. **Use uv run** - Don't call .venv/bin/python directly
4. **Python downloads** - uv can download Python, no need for pyenv

## Summary: Why Switch?

- ⚡ **10-100x faster** than traditional tools
- 🎯 **Single tool** replaces pip, venv, pyenv, pip-tools
- 🔒 **Reproducible** builds by default
- 🚀 **Zero friction** - just `uv sync` and go
- 🐍 **Python management** built-in
- 📦 **Modern** dependency resolution
- 🛠️ **Excellent DX** - made for developers, by developers

## Next Steps

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Try it on a new project: `uv init my-project`
3. Migrate an existing project: Create `pyproject.toml` and run `uv sync`
4. Never look back! 🎉

---

**Remember**: The Python ecosystem finally has a tool that just works. Fast, reliable, and modern. That's uv.