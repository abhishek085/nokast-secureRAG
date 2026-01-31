# Contributing to Nokast-secureRAG

Thank you for your interest in contributing to Nokast-secureRAG! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhishek085/nokast-secureRAG.git
   cd nokast-secureRAG
   ```

2. **Set up the environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install Ollama** (for data generation)
   - Download from https://ollama.ai
   - Pull the required model: `ollama pull kimi-k2.5:cloud`

## Project Structure

```
nokast-secureRAG/
├── src/              # Core source code
│   ├── data_gen.py   # Data generation
│   ├── train_mlx.py  # Model training
│   └── detector.py   # Injection detection
├── eval/             # Evaluation scripts
│   └── benchmark.py  # Benchmarking
├── data/             # Generated datasets (gitignored)
├── models/           # Model checkpoints (gitignored)
└── tests/            # Future: unit tests
```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update README if adding new features

3. **Test your changes**
   ```bash
   python validate.py
   python quickstart.py
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Keep functions focused and modular

## Areas for Contribution

- **Detection Algorithms**: Improve pattern-based detection or add new ML models
- **Benchmarking**: Add new evaluation metrics or test datasets
- **Documentation**: Improve README, add tutorials, create examples
- **Performance**: Optimize inference speed or memory usage
- **Testing**: Add unit tests, integration tests
- **Features**: 
  - Support for additional models
  - Web UI for detection
  - API server
  - Additional attack patterns

## Reporting Issues

When reporting issues, please include:
- Nokast-secureRAG version
- macOS version and chip (M1/M2/M3)
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## Questions?

Feel free to open an issue for discussion before starting major work.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
