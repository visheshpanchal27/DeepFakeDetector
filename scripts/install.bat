@echo off
echo Installing DeepFake Detector dependencies...
echo.

REM Try py launcher first
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using py launcher...
    py -m pip install --upgrade pip
    py -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy pathlib2 Pillow
    echo.
    echo Installation complete! Run: py main.py
    goto :end
)

REM Try python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using python3...
    python3 -m pip install --upgrade pip
    python3 -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy pathlib2 Pillow
    echo.
    echo Installation complete! Run: python3 main.py
    goto :end
)

REM Try python
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using python...
    python -m pip install --upgrade pip
    python -m pip install opencv-python numpy matplotlib seaborn scikit-learn scipy pathlib2 Pillow
    echo.
    echo Installation complete! Run: python main.py
    goto :end
)

echo ERROR: Python not found!
echo Please install Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation

:end
pause