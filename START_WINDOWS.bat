@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
python test_model.py
if errorlevel 1 goto error
python scripts\doctor.py
if errorlevel 1 goto error
echo.
echo Starting the Helmet Violation Detection web app...
echo Open http://127.0.0.1:8000 in your browser. Press Ctrl+C here to stop it.
uvicorn app.main:app --host 127.0.0.1 --port 8000
goto end
:error
echo.
echo Setup or startup failed. Read the error above and check the README (Troubleshooting).
pause
:end
endlocal
