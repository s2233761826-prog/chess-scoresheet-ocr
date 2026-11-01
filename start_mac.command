#!/bin/bash
set -e
cd "$(dirname "$0")"
trap 'echo; echo "启动未完成，请把上方报错复制给我。"; read -r -p "按回车关闭…"' ERR
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "请先安装 Python 3.11，然后重新运行此脚本。"
  open "https://www.python.org/downloads/release/python-3119/"
  exit 1
fi
if [ ! -x .venv-mac/bin/python ]; then
  python3.11 -m venv .venv-mac
fi
if [ ! -f .venv-mac/.setup-complete ]; then
  .venv-mac/bin/python -m pip install --upgrade pip
  .venv-mac/bin/python -m pip install -r requirements.txt
  touch .venv-mac/.setup-complete
fi
.venv-mac/bin/python -m streamlit run app/main.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
