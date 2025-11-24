@echo off
echo ========================================
echo AI Fault Localization Tool - Backend
echo ========================================
echo.

cd backend

echo Checking virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt --quiet

echo.
echo ========================================
echo Starting FastAPI Backend Server
echo ========================================
echo Backend will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000
