@echo off
echo ========================================
echo AI Fault Localization Tool
echo Full Stack Startup
echo ========================================
echo.
echo This will start both backend and frontend servers.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers
echo ========================================
echo.

:: Start backend in a new window
start "AI Fault Localization - Backend" cmd /k "call start_backend.bat"

:: Wait a bit for backend to start
timeout /t 5 /nobreak > nul

:: Start frontend in a new window
start "AI Fault Localization - Frontend" cmd /k "call start_frontend.bat"

echo.
echo Both servers are starting in separate windows.
echo Please wait for them to fully initialize...
echo.
echo Once ready, open http://localhost:3000 in your browser.
echo.
pause
