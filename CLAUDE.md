# Claude Code Development Guidelines

This file contains development guidelines and tool usage for working with this codebase using Claude Code.

## Code Quality & Security Tools

Use these tools for testing, linting, and security analysis:

### Core Linting & Formatting
```bash
# Code formatting and linting
ruff check --fix .          # Fast Python linter with auto-fix
ruff format .               # Code formatting

# Type checking
pyright                     # Static type checker
mypy .                      # Alternative type checker
```

### Code Quality Analysis
```bash
# Dead code detection
vulture .                   # Find unused code

# Import optimization
autoflake --remove-all-unused-imports --recursive .

# Dependency analysis
deptry .                    # Check for unused/missing dependencies

# Code complexity
radon cc --average .        # Cyclomatic complexity
xenon --max-absolute B .    # Complexity monitoring
```

### Documentation Quality
```bash
# Documentation coverage
interrogate .               # Check docstring coverage

# Documentation style
pydocstyle .               # Check docstring conventions

# Documentation generation
mkdocs serve               # Serve documentation locally
mkdocs build               # Build documentation
```

### Security Analysis
```bash
# Security scanning
bandit -r .                # Security issues scanner
semgrep --config=auto .    # Static analysis security tool
detect-secrets scan .      # Find secrets in code
pip-audit                  # Check for vulnerable dependencies
```

### Code Duplication
```bash
# Duplication detection
npx jscpd .                # JavaScript/TypeScript duplication (if Node.js available)
# Alternative: Use ruff for Python-specific duplication rules
```

## Development Workflow

When working on this codebase:

1. **Before making changes:**
   ```bash
   ruff check .
   vulture .
   bandit -r .
   ```

2. **After making changes:**
   ```bash
   ruff check --fix .
   ruff format .
   pyright
   ```

3. **Before committing:**
   ```bash
   # Full quality check
   ruff check .
   vulture .
   deptry .
   bandit -r .
   pip-audit
   ```

4. **For major refactoring:**
   ```bash
   # Complexity analysis
   radon cc --average .
   xenon --max-absolute B .
   
   # Documentation check
   interrogate .
   pydocstyle .
   ```

## Tool Configuration

These tools should be configured in `pyproject.toml` or their respective config files for consistent usage across the team.

### Recommended pyproject.toml section:
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "PL"]

[tool.pyright]
typeCheckingMode = "strict"
pythonVersion = "3.11"

[tool.vulture]
min_confidence = 80
paths = ["src", "tests"]
```

## Security Guidelines

- Always run `bandit` and `semgrep` before committing sensitive changes
- Use `detect-secrets` to prevent credential leaks
- Run `pip-audit` regularly to check for vulnerable dependencies
- Never commit secrets, tokens, or credentials to the repository

## Performance Monitoring

- Use `radon` to monitor cyclomatic complexity
- Keep functions under complexity rating B (moderate complexity)
- Use `vulture` to identify and remove dead code regularly