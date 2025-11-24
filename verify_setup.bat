@echo off
echo ========================================
echo AI Fault Localization - Setup Verification
echo ========================================
echo.

:: Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    goto :error
) else (
    python --version
    echo [OK] Python is installed
)
echo.

:: Check Node.js
echo [2/5] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js 14 or higher.
    echo Download from: https://nodejs.org/
    goto :error
) else (
    node --version
    echo [OK] Node.js is installed
)
echo.

:: Check npm
echo [3/5] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm not found. It should come with Node.js.
    goto :error
) else (
    npm --version
    echo [OK] npm is installed
)
echo.

:: Check Git (optional)
echo [4/5] Checking Git installation (optional)...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Git not found. Some features may not work.
    echo Download from: https://git-scm.com/downloads
) else (
    git --version
    echo [OK] Git is installed
)
echo.

:: Check directory structure
echo [5/5] Checking project structure...
if not exist "backend" (
    echo [ERROR] backend directory not found
    goto :error
)
if not exist "frontend" (
    echo [ERROR] frontend directory not found
    goto :error
)
if not exist "backend\requirements.txt" (
    echo [ERROR] backend\requirements.txt not found
    goto :error
)
if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found
    goto :error
)
echo [OK] Project structure is correct
echo.

:: Summary
echo ========================================
echo Verification Complete!
echo ========================================
echo.
echo All prerequisites are installed.
echo You can now run the application:
echo.
echo Option 1 - Start both servers:
echo   start_all.bat
echo.
echo Option 2 - Start individually:
echo   start_backend.bat  (in one terminal)
echo   start_frontend.bat (in another terminal)
echo.
echo For detailed instructions, see QUICKSTART.md
echo ========================================
pause
exit /b 0

:error
echo.
echo ========================================
echo Verification Failed!
echo ========================================
echo.
echo Please fix the errors above and run this script again.
echo.
pause
exit /b 1
