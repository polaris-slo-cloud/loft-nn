#!/bin/bash

echo "This script will perform the following actions in the current directory:"
echo "1. Create a virtual environment named 'venv'."
echo "2. Install dependencies."
echo "3. Install LoftNN."
echo ""

read -p "Do you want to proceed? (y/n): " confirm

if [[ $confirm == [yY] ]]; then
  echo "Creating virtual environment..."
  python -m venv venv

  source venv/bin/activate

  echo "Installing dependencies..."
  pip install --upgrade pip
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt

  echo "Installing LoftNN..."
  pip install .

  echo "Environment setup complete."

  echo "To activate it, run: source venv/bin/activate"
else
  echo "Setup aborted."
fi
