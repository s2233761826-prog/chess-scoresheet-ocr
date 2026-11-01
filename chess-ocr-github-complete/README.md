# Chess Scoresheet OCR & PGN Correction

国际象棋手写棋谱识别与 PGN 校正系统。通过图片上传、OCR 识别、基于棋规的交互式校正，将纸质棋谱转换为可下载的 PGN。

## 项目背景

原始项目由 Zijie Tang 于 2024 年开发；此源码包整理自家人在 2026 年更新的版本。原云端演示目前不可访问，本仓库用于源码展示与本地运行，不代表在线服务已恢复。

## 功能

- JPG/PNG 图片上传及 EXIF 方向处理。
- PaddleOCR 文字检测、识别、置信度筛选及检测框可视化。
- 使用 python-chess 逐步解析走法、维护棋盘局面。
- 对无法解析或不合法的走法，从当前局面合法走法中按 Levenshtein 距离推荐候选，支持人工输入。
- PGN 预览与下载、修正日志保存。
- 包含检测与识别的数据准备、训练配置和模型导出脚本。

## 本地启动（Apple Silicon Mac）

安装 Python 3.11 后，在项目目录的终端执行：bash start_mac.command

首次启动会创建独立环境并安装依赖。保持终端运行，在浏览器打开终端显示的地址（通常为 http://127.0.0.1:8501）。

## 随附推理模型

本包包含用户提供的检测和识别模型，各自包含三个推理文件：
- models/finetuned/det_chess_infer/
- models/finetuned/rec_chess_infer/
- data/dict_chess.txt：与识别模型配套的字符字典。

同名模型文件分属不同目录，不可互相覆盖。运行推理无需重新训练。
依赖已固定 PyArrow 17.0.0，避免与 NumPy 1.24.3 冲突。

## 本地启动（Windows）

本项目提供的依赖配置面向 Python 3.10 / 3.11。进入项目根目录后运行：

```bat
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
run_app.bat
```

浏览器打开终端显示的 Local URL（通常为 `http://localhost:8501`）。这是本机地址，不是可以放进简历让别人访问的公网网址。

未提供微调模型时，程序会尝试下载并使用默认英文 OCR 模型；默认模型的手写识别效果可能有限。使用微调识别模型时，应同时提供匹配的 `data/dict_chess.txt`。

## 源码结构

- `app/`：界面、OCR 推理、走法校验、PGN 生成。
- `scripts/`：识别与检测的数据准备、训练、导出，以及两个 YAML 配置。
- `requirements.txt`：原项目依赖版本。
- `run_app.bat`：Windows 启动入口。
- `docs/original-setup.md`：保留的原始训练与环境说明。

## 训练入口

识别：`1_prepare_data.py` → `2_train_rec.py` → `3_export_inference.py`。

检测：`1_prepare_det_data.py` → `2_train_det.py` → `4_export_inference_det.py`。

均从项目根目录使用 `python scripts/文件名.py` 执行。训练需自行准备带标注的图片；本包包含上述推理模型，不包含私人训练图片、标注和重新训练所需的完整数据集。原始操作说明见 [original-setup.md](docs/original-setup.md)。

## 验证范围与已知限制

本次整理保留原业务代码，规范目录和主程序文件名，补充说明与忽略规则。Python 文件已通过静态语法解析。用户的 Apple Silicon Mac 已成功安装依赖并启动 Streamlit 服务；尚未确认完整图片识别与 PGN 导出测试，未验证识别准确率。

- 校验走法合法性不能保证识别内容与原棋谱完全一致，仍需人工核对。
- 当前按行、列坐标排序，对多栏棋谱可能需要进一步处理。
- 识别预处理会把开头的小写 b 转为 B，可能混淆 b 兵走法；需人工核对。
- 原识别数据脚本在生成增强样本后随机拆分数据，原样本与增强版本可能分到训练和验证两侧；正式评估前应按原始照片划分，再仅增强训练集。
- 识别配置采用 MobileNetV3/CTC，而下载权重标为 PP-OCRv4，重新训练前需核对架构与权重兼容性。
- 下载地址和依赖版本沿用原项目；不保证所有镜像持续可用。

## 展示链接

创建 GitHub 仓库后，可将仓库页面网址作为简历项目链接。在线体验需要单独部署，GitHub 源码仓库本身不会运行这个 Python OCR 服务。
