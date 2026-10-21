@echo off
echo ========================================================
echo     Starting Smart Number Plate Reader Web UI
echo ========================================================
echo Activating virtual environment and launching application...
cd /d "%~dp0"
call .\.venv\Scripts\activate
streamlit run app.py
pause
