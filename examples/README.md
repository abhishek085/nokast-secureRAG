# Nokast-secureRAG Examples

This directory contains example scripts demonstrating various use cases.

## Available Examples

### 1. Detector Usage (`detector_usage.py`)
Shows how to integrate the prompt injection detector into your RAG application.

```bash
python examples/detector_usage.py
```

**Use Case**: Protecting a production RAG system from prompt injection attacks.

### 2. Custom Dataset Generation (`custom_dataset.py`)
Demonstrates how to create domain-specific training datasets.

```bash
python examples/custom_dataset.py
```

**Use Case**: Adapting the system for specific domains (healthcare, finance, etc.).

## Running Examples

From the project root:

```bash
# Make sure you've set up the environment
python quickstart.py  # First, verify the setup

# Then run individual examples
python examples/detector_usage.py
python examples/custom_dataset.py
```

## Creating Your Own Examples

Feel free to create additional examples and submit them via pull request!
