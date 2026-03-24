#!/bin/bash

# install-pdf-theme.sh
# Script to zip, uninstall, and install the pdf-theme plugin for DITA-OT

set -e  # Exit on error

PLUGIN_ID="pdf-theme"
PLUGIN_DIR="pdf-theme"
ZIP_FILE="pdf-theme.zip"

echo "🔧 Installing pdf-theme plugin to DITA-OT..."

# Check if DITA-OT is available
if ! command -v dita &> /dev/null; then
    echo "❌ Error: DITA-OT command 'dita' not found."
    echo "   Please install DITA-OT and ensure 'dita' is in your PATH."
    echo "   Download: https://www.dita-ot.org/download"
    exit 1
fi

echo "✓ DITA-OT found: $(dita --version | head -n 1)"

# Check if pdf-theme directory exists
if [ ! -d "$PLUGIN_DIR" ]; then
    echo "❌ Error: $PLUGIN_DIR directory not found."
    exit 1
fi

# Step 1: Create zip file
echo "📦 Creating zip file from $PLUGIN_DIR..."
if [ -f "$ZIP_FILE" ]; then
    rm "$ZIP_FILE"
    echo "   Removed existing $ZIP_FILE"
fi

# Create zip (excluding .DS_Store and other hidden files)
cd "$PLUGIN_DIR"
zip -r "../$ZIP_FILE" . -x "*.DS_Store" -x "__MACOSX/*" > /dev/null
cd ..

if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: Failed to create $ZIP_FILE"
    exit 1
fi

echo "✓ Created $ZIP_FILE ($(ls -lh $ZIP_FILE | awk '{print $5}'))"

# Step 2: Uninstall existing plugin (if it exists)
echo "🗑️  Checking for existing $PLUGIN_ID plugin..."

# Check if plugin is installed (matches "pdf-theme" or "pdf-theme@version")
if dita plugins | grep -q "^$PLUGIN_ID"; then
    echo "   Found existing plugin, uninstalling..."
    dita uninstall $PLUGIN_ID
    echo "✓ Uninstalled existing $PLUGIN_ID plugin"
else
    echo "   No existing plugin found"
fi

# Step 3: Install new version
echo "📥 Installing $PLUGIN_ID plugin from $ZIP_FILE..."
dita install "$ZIP_FILE"

# Verify installation
if dita plugins | grep -q "^$PLUGIN_ID"; then
    echo "✅ Successfully installed $PLUGIN_ID plugin"
    echo ""
    echo "You can now use the plugin with:"
    echo "   dita -i yourmap.ditamap -f pdf-theme -o output"
else
    echo "⚠️  Warning: Plugin may not have installed correctly"
    exit 1
fi

echo ""
echo "🎉 Done!"
