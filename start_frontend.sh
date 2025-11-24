#!/bin/bash

echo "========================================"
echo "AI Fault Localization Tool - Frontend"
echo "========================================"
echo ""

cd frontend

echo "Checking node_modules..."
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo ""
echo "========================================"
echo "Starting React Development Server"
echo "========================================"
echo "Frontend will be available at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

npm start
