@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app/main.py
) else (
  echo 未找到 .venv\Scripts\python.exe
  echo 请在项目根目录创建虚拟环境并安装依赖，例如：
  echo   py -3.10 -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
endlocal
