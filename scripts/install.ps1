Write-Host "Installing DeepFake Detector dependencies..." -ForegroundColor Green
Write-Host ""

# Try py launcher first
try {
    py --version | Out-Null
    Write-Host "Using py launcher..." -ForegroundColor Yellow
    py -m pip install --upgrade pip
    py -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy
    Write-Host ""
    Write-Host "Installation complete! Run: py main.py" -ForegroundColor Green
    exit
} catch {
    Write-Host "py launcher not found, trying python3..." -ForegroundColor Yellow
}

# Try python3
try {
    python3 --version | Out-Null
    Write-Host "Using python3..." -ForegroundColor Yellow
    python3 -m pip install --upgrade pip
    python3 -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy
    Write-Host ""
    Write-Host "Installation complete! Run: python3 main.py" -ForegroundColor Green
    exit
} catch {
    Write-Host "python3 not found, trying python..." -ForegroundColor Yellow
}

# Try python
try {
    python --version | Out-Null
    Write-Host "Using python..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy
    Write-Host ""
    Write-Host "Installation complete! Run: python main.py" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
}

Read-Host "Press Enter to continue"