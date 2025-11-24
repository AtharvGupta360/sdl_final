@echo off
echo ========================================
echo AI Fault Localization Tool - Frontend
echo ========================================
echo.

cd frontend

echo Checking node_modules...
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)

echo.
echo ========================================
echo Starting React Development Server
echo ========================================
echo Frontend will be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

call npm start
