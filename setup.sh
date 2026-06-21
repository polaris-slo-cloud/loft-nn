#!/bin/bash

echo "This script will perform the following actions in the current directory:"
echo "1. Create a virtual environment named 'venv' and activate it."
echo "2. Upgrade pip."
echo "3. Install packages from 'requirements.txt'."
echo "4. Install LoftNN."
echo ""

read -p "Do you want to proceed? (y/n): " confirm

if [[ $confirm == [yY] ]]; then
  echo "Creating virtual environment..."
  python -m venv venv

  source venv/bin/activate

  echo "Upgrading pip..."
  pip install --upgrade pip

  echo "Installing requirements..."
  pip install -r requirements.txt

  echo "Installing LoftNN..."
  pip install -e .

  echo "Environment setup complete."

  echo "To activate it, run: source venv/bin/activate"
else
  echo "Setup aborted."
fi
