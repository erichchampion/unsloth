#!/bin/zsh
#
# Build the PDF and the EPUB from md/.
#
# The generators shell out to DITA-OT, which needs a Java 17+ runtime. Neither
# is inherited from an interactive shell here — this script builds its own venv
# and is run non-interactively — so both are resolved explicitly below.
# Override either from the environment:
#
#   DITA_COMMAND=/path/to/dita JAVA_HOME=/path/to/jdk ./publish-book.zsh

set -e

# --- Java -----------------------------------------------------------------
if [[ -z "$JAVA_HOME" ]]; then
    for candidate in \
        /opt/homebrew/opt/openjdk@17 \
        /opt/homebrew/opt/openjdk \
        "$(/usr/libexec/java_home 2>/dev/null)"
    do
        if [[ -n "$candidate" && -x "$candidate/bin/java" ]]; then
            export JAVA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "$JAVA_HOME" ]]; then
    echo "❌ No Java runtime found. DITA-OT 4.4 requires Java 17."
    echo "   brew install openjdk@17    (or set JAVA_HOME yourself)"
    exit 1
fi
echo "✓ JAVA_HOME: $JAVA_HOME"

# --- DITA-OT --------------------------------------------------------------
if [[ -z "$DITA_COMMAND" ]]; then
    if command -v dita > /dev/null 2>&1; then
        DITA_COMMAND="$(command -v dita)"
    elif [[ -x "$HOME/opt/dita/bin/dita" ]]; then
        DITA_COMMAND="$HOME/opt/dita/bin/dita"
    fi
fi

if [[ -z "$DITA_COMMAND" || ! -x "$DITA_COMMAND" ]]; then
    echo "❌ DITA-OT not found. Install it, then set DITA_COMMAND."
    echo "   https://www.dita-ot.org/download"
    exit 1
fi
export DITA_COMMAND
echo "✓ DITA-OT: $DITA_COMMAND ($("$DITA_COMMAND" --version 2>/dev/null | head -n 1))"

# --- Python ---------------------------------------------------------------
# Installing from requirements.txt rather than an inline list: the two used to
# disagree, and the inline one pulled in four packages nothing imports.
python3 -m venv venv --clear
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt

# --- Build ----------------------------------------------------------------
./venv/bin/python generate-dita.py
./venv/bin/python generate-pdf.py
./venv/bin/python generate-epub.py
