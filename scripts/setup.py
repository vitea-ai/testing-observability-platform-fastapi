#!/usr/bin/env python3
"""
Setup script for initializing the FastAPI template project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command: str, check: bool = True) -> bool:
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    
    return True


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    return shutil.which("uv") is not None


def install_uv():
    """Install uv package manager."""
    print("\n📦 Installing uv package manager...")
    
    if sys.platform == "win32":
        print("Please install uv manually from: https://github.com/astral-sh/uv")
        return False
    else:
        command = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        if run_command(command):
            print("✅ uv installed successfully")
            print("Please run: source $HOME/.cargo/env")
            return True
    return False


def setup_environment():
    """Setup environment files."""
    print("\n🔧 Setting up environment...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
    elif env_file.exists():
        print("ℹ️  .env file already exists")
    else:
        print("⚠️  .env.example not found")


def install_dependencies():
    """Install project dependencies."""
    print("\n📚 Installing dependencies...")
    
    if run_command("uv sync"):
        print("✅ Dependencies installed")
        return True
    return False


def setup_git_hooks():
    """Setup git hooks for code quality."""
    print("\n🔗 Setting up git hooks...")
    
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # Create pre-commit hook
        pre_commit = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# Run code formatting and linting before commit
uv run black app/ tests/ --check
uv run ruff check app/ tests/
uv run mypy app/
"""
        pre_commit.write_text(pre_commit_content)
        pre_commit.chmod(0o755)
        print("✅ Git hooks configured")
    else:
        print("ℹ️  Not a git repository, skipping hooks")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "data",
        ".docker",
        "app/api/v1/endpoints",
        "app/core",
        "app/models",
        "app/schemas",
        "app/services",
        "app/utils",
        "tests/unit",
        "tests/integration",
        "docs",
        "scripts",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")


def main():
    """Main setup function."""
    print("🚀 FastAPI Template Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check and install uv
    if not check_uv_installed():
        print("⚠️  uv is not installed")
        if not install_uv():
            print("❌ Failed to install uv")
            sys.exit(1)
    else:
        print("✅ uv is installed")
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup git hooks
    setup_git_hooks()
    
    print("\n" + "=" * 50)
    print("✨ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run: uv run uvicorn app.main:app --reload")
    print("3. Open: http://localhost:8000/docs")
    print("\nHappy coding! 🎉")


if __name__ == "__main__":
    main()
