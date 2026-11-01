#!/bin/bash
set -e
cd "$(dirname "$0")"
python3.11 - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
source=Path.cwd()
target=Path.home()/"Desktop"/"chess-ocr-mac"
if not (target/"app"/"main.py").exists():
    raise SystemExit("没有找到桌面的 chess-ocr-mac/app/main.py。请将当前程序文件夹保持原名放在桌面。")
backup=target/"backups"/("app-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
shutil.copytree(target/"app",backup)
shutil.copytree(target/"models", backup.parent/(backup.name+"-models"))
shutil.copy2(target/"data"/"dict_chess.txt", backup.parent/(backup.name+"-dict.txt"))
shutil.copytree(source/"app",target/"app",dirs_exist_ok=True)
shutil.copytree(source/"models",target/"models",dirs_exist_ok=True)
shutil.copy2(source/"data"/"dict_chess.txt",target/"data"/"dict_chess.txt")
shutil.copy2(source/"start_mac.command",target/"start_mac.command")
for name in ("requirements.txt","README.md","MODEL_REPORT.md","MAC_README.txt"):
    shutil.copy2(source/name,target/name)
shutil.copytree(source/"examples",target/"examples",dirs_exist_ok=True)
print("更新完成；旧 app 已备份到",backup)
PY
bash "$HOME/Desktop/chess-ocr-mac/start_mac.command"
