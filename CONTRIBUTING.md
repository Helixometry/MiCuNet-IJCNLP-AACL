# Contributing to MiCuNet

Thank you for your interest in contributing to MiCuNet! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions with the community.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**
- **Include your environment details** (OS, Python version, PyTorch version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow the Python coding style
- Include appropriate test cases
- Update documentation as needed
- End all files with a newline

## Development Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/MiCuNet-IJCNLP-AACL.git
   cd MiCuNet-IJCNLP-AACL
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

6. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add your meaningful commit message"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a Pull Request on GitHub

## Code Style

We follow PEP 8 with these guidelines:

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Use type hints where appropriate

### Code Quality Tools

Run these before submitting:

```bash
# Format code
black micunet/ scripts/

# Check style
flake8 micunet/ scripts/

# Run tests
pytest tests/
```

## Testing

- Add tests for new features
- Ensure all tests pass before submitting a PR
- Aim for good test coverage
- Use descriptive test names

Run tests with:
```bash
pytest tests/ -v
```

## Documentation

- Update README.md if you change functionality
- Add docstrings to all public functions and classes
- Update relevant documentation in `docs/` folder
- Include examples for new features

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add mixed-curvature projection layer

This implements the Hyperbolic-Euclidean-Spherical projection mechanism
described in the paper. Includes temporal gating for adaptive weighting.

Fixes #123
```

## Recognition

Contributors will be recognized in:
- This file (CONTRIBUTING.md)
- GitHub's contributor graph
- Release notes for major contributions

## Questions?

Feel free to:
- Create a GitHub issue for questions
- Contact us at girish281003@gmail.com
- Check existing documentation in `docs/` folder

Thank you for contributing to MiCuNet! 🎉
