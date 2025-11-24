#!/bin/bash

echo "========================================"
echo "AI Fault Localization - Setup Verification"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error_count=0

# Check Python
echo "[1/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    python3 --version
    echo -e "${GREEN}[OK]${NC} Python is installed"
else
    echo -e "${RED}[ERROR]${NC} Python not found. Please install Python 3.8 or higher."
    echo "Visit: https://www.python.org/downloads/"
    ((error_count++))
fi
echo ""

# Check Node.js
echo "[2/5] Checking Node.js installation..."
if command -v node &> /dev/null; then
    node --version
    echo -e "${GREEN}[OK]${NC} Node.js is installed"
else
    echo -e "${RED}[ERROR]${NC} Node.js not found. Please install Node.js 14 or higher."
    echo "Visit: https://nodejs.org/"
    ((error_count++))
fi
echo ""

# Check npm
echo "[3/5] Checking npm installation..."
if command -v npm &> /dev/null; then
    npm --version
    echo -e "${GREEN}[OK]${NC} npm is installed"
else
    echo -e "${RED}[ERROR]${NC} npm not found. It should come with Node.js."
    ((error_count++))
fi
echo ""

# Check Git (optional)
echo "[4/5] Checking Git installation (optional)..."
if command -v git &> /dev/null; then
    git --version
    echo -e "${GREEN}[OK]${NC} Git is installed"
else
    echo -e "${YELLOW}[WARNING]${NC} Git not found. Some features may not work."
    echo "Visit: https://git-scm.com/downloads"
fi
echo ""

# Check directory structure
echo "[5/5] Checking project structure..."
all_files_exist=true

if [ ! -d "backend" ]; then
    echo -e "${RED}[ERROR]${NC} backend directory not found"
    all_files_exist=false
    ((error_count++))
fi

if [ ! -d "frontend" ]; then
    echo -e "${RED}[ERROR]${NC} frontend directory not found"
    all_files_exist=false
    ((error_count++))
fi

if [ ! -f "backend/requirements.txt" ]; then
    echo -e "${RED}[ERROR]${NC} backend/requirements.txt not found"
    all_files_exist=false
    ((error_count++))
fi

if [ ! -f "frontend/package.json" ]; then
    echo -e "${RED}[ERROR]${NC} frontend/package.json not found"
    all_files_exist=false
    ((error_count++))
fi

if $all_files_exist; then
    echo -e "${GREEN}[OK]${NC} Project structure is correct"
fi
echo ""

# Summary
echo "========================================"
if [ $error_count -eq 0 ]; then
    echo -e "${GREEN}Verification Complete!${NC}"
    echo "========================================"
    echo ""
    echo "All prerequisites are installed."
    echo "You can now run the application:"
    echo ""
    echo "Option 1 - Start backend:"
    echo "  chmod +x start_backend.sh"
    echo "  ./start_backend.sh"
    echo ""
    echo "Option 2 - Start frontend (in another terminal):"
    echo "  chmod +x start_frontend.sh"
    echo "  ./start_frontend.sh"
    echo ""
    echo "For detailed instructions, see QUICKSTART.md"
    echo "========================================"
    exit 0
else
    echo -e "${RED}Verification Failed!${NC}"
    echo "========================================"
    echo ""
    echo "Found $error_count error(s). Please fix them and run this script again."
    echo ""
    exit 1
fi
