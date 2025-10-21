#!/bin/bash
set -e  # Stop if any command fails

echo "=== STEP 1: Updating and installing Python 3.10 ==="
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-distutils python3.10-dev curl git

echo "=== STEP 2: Setting Python 3.10 as default ==="
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --set python /usr/bin/python3.10
sudo update-alternatives --set python3 /usr/bin/python3.10

echo "=== STEP 3: Installing pip for Python 3.10 ==="
curl -sS https://bootstrap.pypa.io/get-pip.py | python

echo "=== STEP 4: Cloning your GitHub repository ==="
cd /content || cd ~
if [ -d "daugsurro" ]; then
  echo "Repository already exists, pulling latest changes..."
  cd daugsurro && git pull
else
  git clone https://github.com/dougdabaso/daugsurro.git
  cd daugsurro
fi

echo "=== STEP 5: Installing dependencies from requirements.txt ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt