@echo off
echo Installing AI Detection System...
echo.

REM Install core dependencies only
python -m pip install --upgrade pip
python -m pip install opencv-python numpy scipy

echo.
echo Installation complete!
echo Run: python main.py
pause