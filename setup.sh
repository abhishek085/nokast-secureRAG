#!/bin/bash
# Setup script for Nokast-secureRAG on macOS

echo "🚀 Setting up Nokast-secureRAG..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This project is optimized for macOS with Apple Silicon"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models/adapters eval

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install from https://ollama.ai"
    echo "   After installation, run: ollama pull kimi-k2.5:cloud"
else
    echo "✅ Ollama found"
    echo "📥 Pulling kimi-k2.5:cloud model..."
    ollama pull kimi-k2.5:cloud
fi

echo ""
echo "✨ Setup complete! Next steps:"
echo ""
echo "1. Generate data:     python src/data_gen.py"
echo "2. Train model:       python src/train_mlx.py"
echo "3. Test detection:    python src/detector.py"
echo "4. Run benchmarks:    python eval/benchmark.py"
echo ""
echo "📖 See README.md for detailed usage instructions"
