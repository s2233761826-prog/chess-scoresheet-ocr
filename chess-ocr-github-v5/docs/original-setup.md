# 国际象棋手写棋谱 OCR 与 PGN 校正系统

本项目使用 **PaddleOCR (PP-OCRv4)** 做棋谱手写走法识别，并用 **python-chess** 进行走棋合法性校验与交互式校正，最终导出标准 **PGN**。

> 说明：你现有的标注文件看起来是 **LabelMe 格式**（`shapes[].label` + `points`），本项目的数据准备脚本已做兼容：  
> - 兼容你描述的简化格式：`[{ "text": "...", "bbox": [x1,y1,x2,y2] }, ...]`  
> - 兼容 LabelMe：`{ "shapes": [{ "label": "...", "points": [[x1,y1],[x2,y2]], "shape_type":"rectangle" }, ...] }`

---

## 1. 环境准备（Windows / CPU）

**必须使用 Python 3.10 或 3.11（64-bit）**（PaddlePaddle/PaddleOCR 目前不支持 Python 3.14，因此用 3.14 会出现 `No matching distribution found for paddlepaddle`）。

```bash
cd /d D:\AI学习\project-chess-ocr
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. 数据放置

项目约定的数据目录如下：

```
data/raw/              # 原始图片（jpg/png）
data/raw_annotations/  # 同名 JSON 标注（.json）
```

### 2.1 你已有的 train/ 数据怎么用？

你目前工作区中已有：

```
train/image/   # 图片
train/lable/   # JSON（注意是 lable 不是 label）
```

你可以不手动复制。`scripts/1_prepare_data.py` 会在 `data/raw` 为空时自动回退读取 `train/image` + `train/lable`。

---

## 3. 生成训练数据（裁剪单步小图 + 列表文件）

```bash
python scripts/1_prepare_data.py
```

运行后将生成：

- `data/prepared/`：裁剪后的单步小图（包含轻量数据增强版本）
- `data/train.txt`：训练列表（图片路径 + \t + 文本）
- `data/eval.txt`：验证列表
- `data/dict_chess.txt`：自定义棋谱字典

可选参数（示例）：

```bash
python scripts/1_prepare_data.py --train_ratio 0.9 --augment_factor 1 --seed 42
```

---

## 4. 微调识别模型（PaddleOCR 训练）

本项目会自动下载并缓存：

- PaddleOCR 官方仓库（用于训练/导出工具脚本）
- PP-OCRv4 英文识别预训练权重（作为微调初始权重）

### 4.1 如果你无法访问 GitHub（重要）
如果你的网络打不开 GitHub（包括常见代理镜像），`scripts/2_train_rec.py` 的“自动下载 PaddleOCR 仓库”会失败。

本项目已将 PaddleOCR 仓库下载地址**优先切换为 GitCode 镜像**（国内更常见可访问）。

如果你仍然下载失败，可以改用 **Gitee 镜像仓库**手动获取 PaddleOCR 源码（任选其一）：

- `https://gitee.com/SWHL/PaddleOCR`
- `https://gitee.com/kmtky/PaddleOCR`
- `https://gitee.com/spring297/PaddleOCR`

推荐方式：用 git 克隆（比浏览器下载更稳定）：

```bash
cd /d D:\AI学习\project-chess-ocr\models\pretrained
git clone -b release/2.7 https://gitee.com/SWHL/PaddleOCR.git PaddleOCR-release-2.7-src
```

然后训练时加参数 `--paddleocr_repo` 指向该目录：

```bash
python scripts/2_train_rec.py --batch_size 32 --epoch 30 --paddleocr_repo models/pretrained/PaddleOCR-release-2.7-src
```

开始训练：

```bash
python scripts/2_train_rec.py
```

默认配置：
- 小学习率 0.0005
- CPU 训练（无 GPU 自动 fallback）
- 输出到 `models/finetuned/rec_chess/`

如果你是纯 CPU 笔记本，建议把 batch_size 调小：

```bash
python scripts/2_train_rec.py --batch_size 32
```

---

## 5. 导出推理模型（用于 App）

```bash
python scripts/3_export_inference.py
```

如果你是用 Gitee 手动克隆的 PaddleOCR 仓库，同样加上：

```bash
python scripts/3_export_inference.py --paddleocr_repo models/pretrained/PaddleOCR-release-2.7-src
```

导出后推理模型位于：

- `models/finetuned/rec_chess_infer/`

检测模型默认使用 PaddleOCR 英文检测（无需微调）。App 会自动下载/缓存检测模型（或你也可手动下载并放到 `models/pretrained/`）。

---

## 6. 启动客户端（Streamlit）

双击：

- `run_app.bat`

或命令行：

```bash
streamlit run app/main.py
```

功能：
- 上传棋谱照片
- 检测 + 识别走法
- 走棋合法性校验
- 对非法走法给出候选（基于 Levenshtein 相似度），用户确认/手动修正
- 输出 PGN，可下载保存
- 保存用户修正日志（用于后续继续增量训练）

---

## 7. 常见问题

### 7.1 没有 GPU 能训练吗？
可以。此项目默认 CPU 训练；但 CPU 训练速度较慢，建议：
- 降低 batch_size（如 16/32）
- 减少 epoch（先 20 跑通流程）

### 7.2 推理加速（OpenVINO / ONNX）
本版本先保证端到端跑通与可维护性。你训练导出完成后，可以再把推理模型转换为 ONNX/OpenVINO 做部署加速（后续我们可以在 `scripts/` 添加转换脚本）。

