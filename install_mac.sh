#!/bin/bash
# LitKit Mac Installation Script
#
# Prerequisites:
#   git clone <repo-url> litkit
#   cd litkit
#
# Usage:
#   ./install_mac.sh
#
# This script:
#   1. Installs uv (if needed)
#   2. Creates a virtual environment in .venv/
#   3. Installs litkit in editable mode
#
set -e

echo "🔬 LitKit Mac Installer"
echo "======================"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only."
    exit 1
fi

# Check/install uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the PATH update
    if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo "❌ uv installation failed. Please install manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "✅ uv installed"
else
    echo "✅ uv already installed"
fi

# Create virtual environment
VENV_DIR=".venv"
if [[ -d "$VENV_DIR" ]]; then
    echo "♻️  Existing venv found, reusing..."
else
    echo "📁 Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install litkit
echo "📥 Installing LitKit and dependencies..."
uv pip install -e "."

# Verify installation
if command -v litkit &> /dev/null || python -c "import litkit" &> /dev/null; then
    echo ""
    echo "✅ LitKit installed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Set up your LLM backend (choose one):"
    echo "     export OPENAI_API_KEY='sk-...'           # OpenAI API"
    echo "     export LITKIT_LLM_ENDPOINT='http://...'  # Local LLM"
    echo ""
    echo "  3. Run LitKit:"
    echo "     litkit --help"
    echo ""
    echo "  4. Quick test (requires workspace with built indices):"
    echo "     litkit --workspace ./workspace query 'What is apoptosis?'"
    echo ""
    echo "See LITKIT_MAC_GUIDE.md for detailed usage instructions."
else
    echo "❌ Installation verification failed."
    exit 1
fi
