#!/bin/zsh

python3 -m venv venv --clear
source venv/bin/activate
#pip install -r requirements.txt
./venv/bin/pip install -q requests beautifulsoup4 html2text markdown ollama weasyprint PyPDF2 PyYAML

./generate-dita.py
./generate-pdf.py
./generate-epub.py
