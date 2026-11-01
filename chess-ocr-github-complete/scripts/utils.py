"""
通用工具函数

本文件会被数据准备 / 训练 / 推理模块复用，尽量把：
- 路径处理
- JSON 标注解析（兼容 LabelMe 与简化格式）
- bbox 规范化与排序
- 简单图像增强（CPU 友好）
- 文件下载与解压（用于自动拉取 PaddleOCR 仓库与预训练权重）
集中在一起，便于维护。
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OcrItem:
    """单个 OCR 样本（文本 + 矩形 bbox）。"""

    text: str
    # 统一成 [x1, y1, x2, y2]，且保证 x1<=x2, y1<=y2
    bbox: Tuple[int, int, int, int]


def project_root() -> Path:
    """返回项目根目录（假定本文件位于 <root>/scripts/utils.py）。"""
    return Path(__file__).resolve().parents[1]


def find_input_dirs(root: Path) -> Tuple[Path, Path]:
    """
    决定棋谱图片与 JSON 标注目录。
    优先：data/raw + data/raw_annotations
    兜底：train/image + train/lable（注意目录名为 lable）
    """
    raw_dir = root / "data" / "raw"
    ann_dir = root / "data" / "raw_annotations"

    has_raw = raw_dir.exists() and (
        any(raw_dir.glob("*.jpg"))
        or any(raw_dir.glob("*.jpeg"))
        or any(raw_dir.glob("*.png"))
        or any(raw_dir.glob("*.bmp"))
        or any(raw_dir.glob("*.webp"))
    )
    has_ann = ann_dir.exists() and any(ann_dir.glob("*.json"))
    if has_raw and has_ann:
        return raw_dir, ann_dir

    train_img = root / "train" / "image"
    train_ann = root / "train" / "lable"
    if train_img.exists() and train_ann.exists():
        return train_img, train_ann

    return raw_dir, ann_dir


def list_images(img_dir: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    imgs: List[Path] = []
    for e in exts:
        imgs.extend(sorted(img_dir.glob(e)))
    return imgs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def imread_bgr_unicode(path: Path) -> np.ndarray | None:
    """
    兼容 Windows 下包含中文/非 ASCII 路径的图片读取。

    背景：
    - OpenCV 的 cv2.imread 在部分 Windows 构建/环境中对非 ASCII 路径支持不稳定
    - 你当前项目路径包含中文（例如 `D:\\AI学习\\...`），会导致 imread 全部失败

    解决：
    - 使用 numpy.fromfile 读取原始字节，再用 cv2.imdecode 解码
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_jpg_unicode(path: Path, bgr: np.ndarray, quality: int = 95) -> bool:
    """
    兼容 Windows 下包含中文/非 ASCII 路径的图片写入（JPEG）。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False

def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def normalize_bbox_xyxy(
    bbox: Sequence[float],
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    """
    将 bbox 规范化为 (x1, y1, x2, y2) 的整数坐标，并裁剪到图像边界内。

    bbox 允许输入：
    - [x1,y1,x2,y2]
    - [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]（四点）
    - LabelMe rectangle points: [[x1,y1],[x2,y2]]
    """
    if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    else:
        # 解析点集
        pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)
        x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    x1i = _clip_int(x1, 0, w - 1)
    y1i = _clip_int(y1, 0, h - 1)
    x2i = _clip_int(x2, 0, w - 1)
    y2i = _clip_int(y2, 0, h - 1)

    # 防止出现空框
    if x2i <= x1i:
        x2i = min(w - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(h - 1, y1i + 1)
    return (x1i, y1i, x2i, y2i)


def parse_annotation_items(
    ann: Any,
    image_w: int,
    image_h: int,
) -> List[OcrItem]:
    """
    兼容两种标注格式：

    1) 简化格式（你在需求中假设的）：
       [
         {"text": "e4", "bbox": [x1,y1,x2,y2]},
         ...
       ]

    2) LabelMe（你现有数据）：
       {"shapes":[{"label":"e4","points":[[x1,y1],[x2,y2]],"shape_type":"rectangle"}, ...]}
    """
    items: List[OcrItem] = []

    if isinstance(ann, list):
        # 简化格式
        for obj in ann:
            if not isinstance(obj, dict):
                continue
            text = str(obj.get("text", "")).strip()
            bbox = obj.get("bbox", None)
            if not text or bbox is None:
                continue
            x1, y1, x2, y2 = normalize_bbox_xyxy(bbox, image_w, image_h)
            items.append(OcrItem(text=text, bbox=(x1, y1, x2, y2)))
        return items

    if isinstance(ann, dict) and "shapes" in ann:
        # LabelMe
        shapes = ann.get("shapes") or []
        for s in shapes:
            if not isinstance(s, dict):
                continue
            text = str(s.get("label", "")).strip()
            points = s.get("points", None)
            if not text or points is None:
                continue
            x1, y1, x2, y2 = normalize_bbox_xyxy(points, image_w, image_h)
            items.append(OcrItem(text=text, bbox=(x1, y1, x2, y2)))
        return items

    # 未知格式：返回空
    return items


def sort_items_reading_order(items: List[OcrItem], y_tol: int = 20) -> List[OcrItem]:
    """
    将 bbox 按“从上到下、从左到右”排序。

    关键点：同一行的 y 可能有抖动，因此使用 y_tol 将条目分到行里再排序。
    """
    if not items:
        return []

    # 以 bbox 中心点作为排序依据更稳一些
    centers = []
    for it in items:
        x1, y1, x2, y2 = it.bbox
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    idxs = list(range(len(items)))
    idxs.sort(key=lambda i: centers[i][1])  # 先按 y 排

    rows: List[List[int]] = []
    cur: List[int] = []
    cur_y: Optional[float] = None

    for i in idxs:
        y = centers[i][1]
        if cur_y is None:
            cur_y = y
            cur = [i]
            continue
        if abs(y - cur_y) <= y_tol:
            cur.append(i)
            # 行中心做滑动更新，避免越分越偏
            cur_y = (cur_y * (len(cur) - 1) + y) / len(cur)
        else:
            rows.append(cur)
            cur = [i]
            cur_y = y
    if cur:
        rows.append(cur)

    # 行内按 x 排
    sorted_items: List[OcrItem] = []
    for r in rows:
        r.sort(key=lambda i: centers[i][0])
        sorted_items.extend(items[i] for i in r)
    return sorted_items


def preprocess_crop_for_rec(crop_bgr: np.ndarray) -> np.ndarray:
    """
    对裁剪图做轻量预处理，提高手写识别鲁棒性。

    注意：这是数据准备阶段的“图片质量增强”，不是模型输入的强制格式。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 对比度增强（CLAHE 对拍照阴影更友好）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 轻微去噪，避免把细笔画抹掉
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 回到 3 通道（PaddleOCR 训练链通常读彩色）
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def augment_crop(
    crop_bgr: np.ndarray,
    rng: random.Random,
    max_rotate_deg: float = 3.5,
    add_noise: bool = True,
    add_blur: bool = True,
) -> np.ndarray:
    """
    简单数据增强（CPU 友好），面向棋谱裁剪条带：
    - 小角度旋转（拍摄歪斜）
    - 亮度/对比度、Gamma（光照与纸色）
    - 轻微水平缩放（取景透视近似）
    - 高斯模糊（虚焦）、高斯噪声（感光/压缩）

    设计原则：增强偏温和，避免破坏 SAN 笔画结构。
    """
    img = crop_bgr.copy()
    h, w = img.shape[:2]

    # 亮度 / 对比（模拟灯光、深色纸与浅色纸）
    if rng.random() < 0.5:
        alpha = rng.uniform(0.82, 1.18)
        beta = rng.uniform(-28.0, 28.0)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gamma（整体偏亮/偏暗）
    if rng.random() < 0.38:
        gamma = rng.uniform(0.82, 1.14)
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)
        b, g, r = cv2.split(img)
        b = cv2.LUT(b, y)
        g = cv2.LUT(g, y)
        r = cv2.LUT(r, y)
        img = cv2.merge([b, g, r])

    # 旋转
    angle = rng.uniform(-max_rotate_deg, max_rotate_deg)
    if abs(angle) > 1e-3:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 轻微水平缩放（透视/远近）
    if rng.random() < 0.22:
        sx = rng.uniform(0.94, 1.06)
        nw = max(8, int(round(w * sx)))
        img = cv2.resize(img, (nw, h), interpolation=cv2.INTER_LINEAR)
        if nw != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 模糊
    if add_blur and rng.random() < 0.38:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0)

    # 噪声
    if add_noise and rng.random() < 0.52:
        sigma = rng.uniform(3.0, 9.0)
        n = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    return img


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def safe_copy_if_missing(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def download_file(
    url: str,
    dst: Path,
    chunk_size: int = 1 << 20,
    timeout_sec: int = 60,
    retries: int = 5,
    backoff_sec: float = 2.0,
) -> None:
    """
    纯标准库下载（避免额外依赖）。
    若已存在则跳过。
    """
    import urllib.request
    import urllib.error
    import http.client

    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[download] cache hit: {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
        return

    last_err: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

        try:
            print(f"[download] ({attempt}/{retries}) {url} -> {dst}")
            req = urllib.request.Request(url, headers={"User-Agent": "project-chess-ocr/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp, tmp.open("wb") as f:
                total = resp.headers.get("Content-Length")
                total_mb = (int(total) / 1024 / 1024) if total and total.isdigit() else None
                if total_mb:
                    print(f"[download] size: {total_mb:.1f} MB")
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    f.write(buf)
            tmp.replace(dst)
            print(f"[download] done: {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.RemoteDisconnected, TimeoutError) as e:
            last_err = e
            # 指数退避，避免频繁打满网络
            sleep_s = backoff_sec * attempt
            print(f"[download] failed: {type(e).__name__}: {e} (sleep {sleep_s:.1f}s then retry)")
            time.sleep(sleep_s)

    raise RuntimeError(f"下载失败：{url} -> {dst}，已重试 {retries} 次。最后错误：{last_err}")


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)


def download_and_extract_zip(url: str, dst_dir: Path, cache_name: str) -> Path:
    """
    下载 zip 并解压到 dst_dir，返回解压后的根目录（最佳猜测）。
    """
    cache_dir = dst_dir
    ensure_dir(cache_dir)

    zip_path = cache_dir / cache_name
    download_file(url, zip_path)

    # 解压到一个同名目录，避免污染
    out_dir = cache_dir / (Path(cache_name).stem)
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir

    extract_zip(zip_path, out_dir)
    return out_dir


def guess_paddleocr_repo_root(extracted_dir: Path) -> Path:
    """
    PaddleOCR zip 解压后通常会多一层目录（例如 PaddleOCR-release-2.7）。
    这里做一个“找 tools/train.py 的上层目录”的探测。
    """
    candidates = [extracted_dir]
    # 最多下探两层
    for _ in range(2):
        new: List[Path] = []
        for c in candidates:
            if (c / "tools" / "train.py").exists():
                return c
            for child in c.iterdir():
                if child.is_dir():
                    new.append(child)
        candidates = new

    # 兜底：返回原目录
    return extracted_dir

