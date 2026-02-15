# Contributing to KernelForge

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev,test]"
```

This installs:
- **black**: Code formatter
- **isort**: Import sorter
- **ruff**: Fast Python linter
- **mypy**: Static type checker
- **pre-commit**: Git hook manager
- **pytest**: Testing framework

### Setup Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit:

```bash
pre-commit install
```

## Code Quality Tools

### Format Code

Run Black and isort to format your code:

```bash
black python/ tests/
isort python/ tests/
```

### Lint Code

Check for code quality issues with Ruff:

```bash
ruff check python/ tests/
```

Auto-fix issues when possible:

```bash
ruff check --fix python/ tests/
```

### Type Check

Verify type annotations with mypy:

```bash
mypy python/ tests/
```

### Run All Checks

```bash
# Format
black python/ tests/
isort python/ tests/

# Lint
ruff check python/ tests/

# Type check
mypy python/ tests/

# Run tests
pytest
```

## Code Style Guidelines

### General Rules
- **Line length**: 100 characters
- **Python version**: 3.11+
- **Type hints**: Required for all function signatures
- **Docstrings**: Brief 1-2 line descriptions for functions

### Comments
- Use `#` for single-line comments
- Keep comments concise and clear
- Explain *why*, not *what*

### Variable Naming
- **Standard convention**: lowercase with underscores (`my_variable`)
- **Exception**: Scientific code may use uppercase for mathematical notation (X, R, N, Q, K)
  - These are allowed in `tests/` and CLI code
  - Represents common ML/scientific conventions (feature matrices, coordinates, etc.)

### Import Organization
Imports are automatically organized by isort in the following order:
1. Standard library imports
2. Third-party imports
3. Local/project imports

## Testing

Run tests with pytest:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_kernels.py

# With coverage
pytest --cov=kernelforge

# Parallel execution
pytest -n auto
```

## Continuous Integration

GitHub Actions automatically runs on all pull requests:
- Code formatting check (Black)
- Import sorting check (isort)
- Linting (Ruff)
- Type checking (mypy)

Ensure all checks pass before submitting your PR.

## Pre-commit Hooks

When you commit, pre-commit automatically runs:
1. Trailing whitespace removal
2. End-of-file fixer
3. YAML validation
4. Black formatting
5. isort import sorting
6. Ruff linting
7. mypy type checking

If any check fails, the commit is rejected. Fix the issues and commit again.

## Versioning

KernelForge uses **setuptools-scm** for automatic version management based on git tags.

### Version Format

- **Release versions**: Determined by git tags (e.g., `v0.1.14` → version `0.1.14`)
- **Development versions**: Automatic format `X.Y.Z.devN+g<commit>` between releases
- **Example**: `0.1.15.dev7+gabc1234` means 7 commits after v0.1.14 tag

### Creating a New Release

```bash
# 1. Ensure you're on the main branch and up to date
git checkout master
git pull

# 2. Create and push a new tag (with 'v' prefix)
git tag -a v0.1.15 -m "Release version 0.1.15"
git push origin v0.1.15

# 3. Version is automatically set to 0.1.15 when building
```

### Version Detection

The version is automatically detected from git tags during:
- Package building
- Installation
- Import (`from kernelforge import __version__`)

**Note:** You should never manually edit version numbers in the code. Git tags are the single source of truth.

## Questions?

Open an issue on GitHub if you have questions or need help!
