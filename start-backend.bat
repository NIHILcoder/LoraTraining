@echo off
echo ==============================================
echo Installing Python Backend dependencies...
echo ==============================================
cd backend
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt --default-timeout=1000

echo.
echo ==============================================
echo Starting FastAPI Server...
echo ==============================================
uvicorn main:app --reload
