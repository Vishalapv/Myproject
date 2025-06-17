@echo off
setlocal enabledelayedexpansion

echo Property Price Prediction and Market Analysis System
echo ===================================================
echo.

REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b
)

REM Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set "python_version=%%a"
for /f "tokens=1 delims=." %%a in ("!python_version!") do set "major=%%a"
for /f "tokens=2 delims=." %%a in ("!python_version!") do set "minor=%%a"

if !major! lss 3 (
    echo Python version must be 3.8 or higher
    echo Current version: !python_version!
    pause
    exit /b
)

if !major! equ 3 if !minor! lss 8 (
    echo Python version must be 3.8 or higher
    echo Current version: !python_version!
    pause
    exit /b
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b
    )
    echo Virtual environment created successfully
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b
)

REM Install/update requirements
echo Installing/updating requirements...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip
    pause
    exit /b
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b
)
echo Requirements installed successfully
echo.

REM Check if data file exists
if not exist "sample_properties.csv" (
    echo Error: sample_properties.csv not found
    echo Please ensure the data file is in the same directory as this script
    pause
    exit /b
)

REM Force retrain model to ensure compatibility
echo Training new model...
del model.joblib 2>nul
python -c "from app import load_data, train_model; df = load_data(); model = train_model(df)"
if %errorlevel% neq 0 (
    echo Failed to train model
    pause
    exit /b
)
echo Model trained successfully
echo.

REM Start the Flask application
echo Starting Flask application...
echo.
echo The application will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

REM Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=1

REM Start Flask using python directly
python app.py
if %errorlevel% neq 0 (
    echo.
    echo Failed to start Flask application
    echo Please check the error message above
    pause
    exit /b
)

REM Deactivate virtual environment when done
deactivate
pause 