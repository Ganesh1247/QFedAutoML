@echo off
title QFedAutoML Full-Stack Platform Launcher
echo ========================================================
echo   Launching QFedAutoML Real Machine Learning Platform
echo ========================================================
echo.

echo [1/3] Checking SQLite Database initialization...
python -c "from backend.database.connection import init_db; from backend.database.models_orm import *; init_db()"
echo Database is ready!
echo.

echo [2/3] Starting Python FastAPI Backend Server on port 8000...
start "QFedAutoML Backend (Port 8000)" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
echo Backend server started in a new terminal window.
echo.

echo [3/3] Starting Frontend Web Interface on port 3000...
cd frontend
start "QFedAutoML Frontend (Port 3000)" cmd /k "npm run dev"
cd ..
echo Frontend server started in a new terminal window.
echo.

echo ========================================================
echo   QFedAutoML is now running!
echo   Frontend Dashboard: http://localhost:3000
echo   Backend API & Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo.
timeout /t 3 >nul
start http://localhost:3000
exit
