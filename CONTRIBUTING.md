# Contributing to StatsPAI

We welcome contributions to StatsPAI! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new econometric methods or improvements
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve docs, examples, or tutorials
5. **Testing**: Add test cases or improve test coverage

> **On credit**: every merged PR is credited in `CONTRIBUTORS.md` and
> the paper's Acknowledgments. Paper **authorship** is separate —
> extended by invitation at the maintainers' discretion. See
> [Authorship & Acknowledgments](#-authorship--acknowledgments) below.

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/brycewang-stanford/StatsPAI.git
   cd StatsPAI
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

## 📝 Development Workflow

### Before Making Changes

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for major changes to discuss the approach
3. **Read the documentation** to understand the codebase structure

### Making Changes

1. **Write Tests First** (TDD approach recommended)
   ```bash
   # Create test file
   touch tests/test_your_feature.py
   
   # Write failing tests
   pytest tests/test_your_feature.py
   ```

2. **Implement Your Changes**
   - Follow existing code style and patterns
   - Add type hints for all function signatures
   - Include docstrings for public functions
   - Add inline comments for complex logic

3. **Run Tests**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=src/statspai
   
   # Run specific tests
   pytest tests/test_your_feature.py -v
   ```

4. **Check Code Quality**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Check linting
   flake8 src/ tests/
   
   # Type checking (if mypy is configured)
   mypy src/
   ```

### Commit Guidelines

Use conventional commits format:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(causal): add bootstrap confidence intervals to CausalForest"
git commit -m "fix(outreg2): handle empty model lists gracefully"
git commit -m "docs(readme): update installation instructions"
```

## 🏗 Code Structure

### Package Organization
```
src/statspai/
├── __init__.py          # Main API exports
├── core/                # Core regression functionality
│   ├── __init__.py
│   ├── base.py          # Base classes
│   └── regression.py    # Main regression implementation
├── causal/              # Causal inference methods
│   ├── __init__.py
│   └── causal_forest.py # Causal Forest implementation
└── output/              # Output and formatting
    ├── __init__.py
    └── outreg2.py       # Excel export functionality
```

### Code Style Guidelines

1. **Follow PEP 8** with line length of 88 characters
2. **Use type hints** for all function parameters and return values
3. **Write docstrings** in Google format:
   ```python
   def function_name(param1: int, param2: str) -> bool:
       """Brief description of the function.
       
       Args:
           param1: Description of param1.
           param2: Description of param2.
           
       Returns:
           Description of return value.
           
       Raises:
           ValueError: When param1 is negative.
       """
   ```

4. **Use descriptive variable names**
5. **Add comments for complex algorithms**

### Testing Guidelines

1. **Test Coverage**: Aim for >90% test coverage
2. **Test Types**:
   - Unit tests for individual functions
   - Integration tests for workflows
   - Regression tests against known results
3. **Test Structure**:
   ```python
   def test_function_name_scenario():
       # Arrange
       data = create_test_data()
       
       # Act
       result = function_to_test(data)
       
       # Assert
       assert result.some_property == expected_value
   ```

4. **Use fixtures** for common test data:
   ```python
   @pytest.fixture
   def sample_data():
       return pd.DataFrame({
           'y': [1, 2, 3, 4, 5],
           'x': [2, 4, 6, 8, 10]
       })
   ```

## 🧾 Authorship & Acknowledgments

StatsPAI is an academic project that publishes a JOSS paper and
follow-on methodological work. To keep expectations transparent, we
separate **acknowledgment** from **paper authorship**.

### Acknowledgments (automatic)

Everyone whose pull request is merged into `main` is automatically
credited. Specifically:

- Added to `CONTRIBUTORS.md` (name + GitHub handle + area of contribution).
- Listed in the **Acknowledgments** section of the JOSS paper and any
  subsequent methodological papers that build on the codebase they
  touched.
- Credited in release notes (`CHANGELOG.md`) for the version that
  ships their change.

No invitation is needed. This is the default and applies to all merged
contributions — bug fixes, docs, tests, refactors, new estimators.

### Paper Authorship (by invitation)

**By default, code contributors are not co-authors of StatsPAI
papers.** The core maintainers reserve the sole right to extend
authorship invitations to contributors whose work they judge to be
exceptional. There is no fixed rubric — the decision rests with the
maintainers and is made case by case.

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues or private communication

## 📄 License

By contributing to StatsPAI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to StatsPAI! 🎉
