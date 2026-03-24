#!/usr/bin/env bash
#
# Installation script for custom PDF syntax highlighting plugin
#
# Usage: ./install-syntax-highlighting.sh
#

set -e

echo "🎨 Installing PDF Syntax Highlighting Plugin..."
echo

# Check if dita command is available
if ! command -v dita &> /dev/null; then
    echo "❌ Error: 'dita' command not found"
    echo "   Please ensure DITA-OT is installed and in your PATH"
    exit 1
fi

# Get DITA-OT version
DITA_VERSION=$(dita --version 2>&1 | head -1)
echo "📦 Found: $DITA_VERSION"
echo

# Check if plugin directory exists
if [ ! -d "pdf-theme-syntax-highlight" ]; then
    echo "❌ Error: pdf-theme-syntax-highlight directory not found"
    echo "   Please run this script from the book directory"
    exit 1
fi

# Create zip if it doesn't exist
if [ ! -f "pdf-theme-syntax-highlight.zip" ]; then
    echo "📦 Creating plugin archive..."
    cd pdf-theme-syntax-highlight
    zip -q -r ../pdf-theme-syntax-highlight.zip .
    cd ..
fi

# Install the plugin
echo "📥 Installing plugin..."
dita install pdf-theme-syntax-highlight.zip

echo
echo "✅ Installation complete!"
echo

# Verify installation
if dita plugins | grep -q "com.custom.pdf.syntax.highlight"; then
    echo "✓ Plugin successfully installed: com.custom.pdf.syntax.highlight"
    echo
    echo "📚 Usage:"
    echo "   ./generate-pdf.py"
    echo
    echo "   Your PDF will now have syntax-highlighted code blocks!"
    echo "   Supported languages: TypeScript, JavaScript, Bash, JSON"
    echo
else
    echo "⚠️  Warning: Plugin installed but not showing in plugin list"
    echo "   Try running: dita plugins"
fi
