@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul || (
  echo Python is required but not on PATH.
  pause
  exit /b 1
)

python -c "import PIL, numpy" >nul 2>nul
if errorlevel 1 (
  echo Missing dependency ^(Pillow or numpy^).
  echo Install with:  pip install pillow numpy
  pause
  exit /b 1
)

echo Starting NetraXAI prototype...
python server.py %*
pause