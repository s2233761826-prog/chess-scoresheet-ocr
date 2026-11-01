"""
OCR 推理封装（检测 + 识别）

实现目标：
- 检测：默认 Paddle 英文检测；若存在 models/finetuned/det_chess_infer（见 scripts/2_train_det.py）
  则加载微调后的 DB 检测模型（棋谱场景推荐）。
- 识别：若存在 models/finetuned/rec_chess_infer 则加载微调识别模型，否则英文默认识别。
- 返回按阅读顺序排序后的结果：[{ "text": str, "bbox": [x1,y1,x2,y2] }, ...]

注意：
- PaddleOCR 的高层 API `PaddleOCR` 可以同时做 det+rec。
- 我们为了便于替换模型，允许分别指定 det_model_dir / rec_model_dir。
- 若你还没训练/导出识别模型，本模块也允许使用 PaddleOCR 默认英文识别模型先跑通流程。
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from paddleocr import PaddleOCR
import re


def load_rgb_image(path: str) -> Image.Image:
    """
    读取 RGB 图像并按 EXIF Orientation 转正像素方向。
    避免手机竖拍等场景：浏览器/相册显示正常，但 cv2.imread 忽略 EXIF 导致 OCR 框错位。
    """
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


# 过小文件往往是未写完/截断的导出，会在 load_combine 阶段报 partial data
_REC_INFER_MIN_BYTES = 1024


def _infer_model_dir_ready(infer_dir: Path) -> bool:
    """Paddle 推理需要成对的 pdmodel + pdiparams，且大小合理。"""
    d = infer_dir.resolve()
    pdmodel = d / "inference.pdmodel"
    pdiparams = d / "inference.pdiparams"
    if not (pdmodel.is_file() and pdiparams.is_file()):
        return False
    try:
        if (
            pdmodel.stat().st_size < _REC_INFER_MIN_BYTES
            or pdiparams.stat().st_size < _REC_INFER_MIN_BYTES
        ):
            return False
    except OSError:
        return False
    return True


def _clear_rec_ascii_cache(rec_infer_dir: Path) -> None:
    """加载失败时删掉 ASCII 缓存目录，避免长期沿用损坏副本。"""
    key = hashlib.sha1(str(rec_infer_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    local = os.environ.get("LOCALAPPDATA", "")
    root = Path(local) if local else Path.home() / "AppData" / "Local"
    base = root / "project-chess-ocr-cache" / "rec_infer" / key
    if base.is_dir():
        shutil.rmtree(base, ignore_errors=True)


def _paddle_fs_path(p: Path) -> str:
    """
    Paddle 推理在 Windows 上对含非 ASCII 的路径可能无法打开文件；尽量转为 8.3 短路径。
    """
    s = str(p.resolve())
    if os.name != "nt":
        return s
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))
        if n and n < len(buf):
            return buf.value or s
    except Exception:
        pass
    return s


def _materialize_rec_to_ascii_cache(rec_infer_dir: Path) -> str:
    """
    将推理权重复制到 %LOCALAPPDATA%\\project-chess-ocr-cache\\...（一般为纯 ASCII），
    规避 Paddle 推理在部分环境下无法用 fopen 打开含中文路径的问题。
    """
    key = hashlib.sha1(str(rec_infer_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    local = os.environ.get("LOCALAPPDATA", "")
    root = Path(local) if local else Path.home() / "AppData" / "Local"
    base = root / "project-chess-ocr-cache" / "rec_infer" / key
    base.mkdir(parents=True, exist_ok=True)
    for fname in ("inference.pdmodel", "inference.pdiparams"):
        src = rec_infer_dir / fname
        dst = base / fname
        need = not dst.is_file()
        if not need:
            try:
                need = dst.stat().st_mtime < src.stat().st_mtime
            except OSError:
                need = True
        if need:
            shutil.copy2(src, dst)
    return str(base.resolve())


def _ensure_paddle_readable_rec_dir(rec_infer_dir: Path) -> str:
    resolved = rec_infer_dir.resolve()
    s = _paddle_fs_path(resolved)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        return _materialize_rec_to_ascii_cache(resolved)


def _xyxy_from_poly(poly: List[List[float]]) -> List[int]:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def normalize_ocr_text(s: str) -> str:
    """
    OCR 结果的后处理：
    - 去除空格
    - 统一大写（坐标字母 a-h 不强制大写；SAN 中棋子字母大写更常见）
    - 纠正常见 0/O 混淆与王车易位格式
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    s = s.replace(" ", "")
    s = s.replace("×", "x")
    s = s.replace("–", "-").replace("—", "-")

    # 王车易位常见：0-0 / 0-0-0
    s = s.replace("o-o-o", "O-O-O").replace("o-o", "O-O")
    s = s.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    if "O-O" in s:
        s = s.replace("0", "O")

    # 只把“棋子字母”规范成大写；文件字母 a-h 必须保持小写，否则如 Nf3 会变成 NF3 导致走法不匹配
    if s and s[0] in "kqrnp":
        s = s[0].upper() + s[1:]

    # 将 A-H（多为文件字母误大写）降为 a-h，但保留棋子字母与 O
    out = []
    for ch in s:
        if ch in "KQRBNPO":
            out.append(ch)
        elif ch in "ABCDEFGH":
            out.append(ch.lower())
        else:
            out.append(ch)
    s = "".join(out)

    return s


def sort_by_bbox(items: List[Dict[str, Any]], y_tol: int = 25) -> List[Dict[str, Any]]:
    """
    按阅读顺序排序：先按 y 分行，再按 x。
    """
    if not items:
        return []

    centers = []
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    idxs = list(range(len(items)))
    idxs.sort(key=lambda i: centers[i][1])

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
            cur_y = (cur_y * (len(cur) - 1) + y) / len(cur)
        else:
            rows.append(cur)
            cur = [i]
            cur_y = y
    if cur:
        rows.append(cur)

    out: List[Dict[str, Any]] = []
    for r in rows:
        r.sort(key=lambda i: centers[i][0])
        out.extend(items[i] for i in r)
    return out


_SAN_LIKE_RE = re.compile(
    r"^("
    r"O-O(-O)?"
    r"|"
    r"([KQRBN])?"  # piece (pawn omitted)
    r"([a-h])?"  # disambiguation file
    r"([1-8])?"  # disambiguation rank
    r"(x)?"  # capture
    r"([a-h][1-8])"  # target square
    r"(=([QRBN]))?"  # promotion
    r"([+#])?"  # check / mate
    r")$"
)


def is_san_like(s: str) -> bool:
    """
    快速过滤明显不可能的“走法文本”，避免 det 噪声把列表淹没。
    这不是严格 SAN 校验（严格校验交给 python-chess），但能去掉很多垃圾。
    """
    if not s:
        return False
    s = str(s).strip()
    if not s:
        return False
    # 常见 OCR 噪声：单个字母/数字
    if len(s) <= 1:
        return False
    # 统一一些符号后再判断
    s = normalize_ocr_text(s)
    # 只保留 ASCII 可见字符（去掉奇怪控制符）
    if any(ord(ch) < 32 for ch in s):
        return False
    return _SAN_LIKE_RE.match(s) is not None


def looks_like_round_index(text: str) -> bool:
    """
    回合序号常见形式："1", "12.", "12", "12..."（不强依赖 OCR 是否带点）
    """
    if not text:
        return False
    s = str(text).strip()
    if not s:
        return False
    s = s.replace(" ", "")
    # "12." -> "12"；但不能简单 rstrip('.')，否则 "e4." 会被误判成序号 "e4"
    if len(s) >= 2 and s[-1] in ".）)" and s[:-1].isdigit():
        s = s[:-1]
    if s.isdigit():
        return 1 <= len(s) <= 3
    return False


@dataclass
class OcrPredictorConfig:
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    rec_char_dict_path: Optional[str] = None
    use_angle_cls: bool = False
    lang: str = "en"
    det_limit_side_len: Optional[float] = None
    det_limit_type: Optional[str] = None
    det_db_thresh: Optional[float] = None
    det_db_box_thresh: Optional[float] = None
    det_db_unclip_ratio: Optional[float] = None


class OcrPredictor:
    def __init__(self, cfg: OcrPredictorConfig):
        """
        PaddleOCR 初始化会在首次运行时自动下载缺失的预训练模型（如果没指定本地路径）。
        """
        kwargs: Dict[str, Any] = {
            "use_angle_cls": cfg.use_angle_cls,
            "lang": cfg.lang,
            "show_log": False,
            "use_gpu": False,
            "enable_mkldnn": False,
            "drop_score": 0.0,
            "rec_image_shape": "3,48,320",
            "rec_algorithm": "CRNN",
        }
        if cfg.det_model_dir:
            kwargs["det_model_dir"] = cfg.det_model_dir
        if cfg.rec_model_dir:
            kwargs["rec_model_dir"] = cfg.rec_model_dir
        if cfg.rec_char_dict_path:
            kwargs["rec_char_dict_path"] = cfg.rec_char_dict_path
        if cfg.det_limit_side_len is not None:
            kwargs["det_limit_side_len"] = float(cfg.det_limit_side_len)
        if cfg.det_limit_type is not None:
            kwargs["det_limit_type"] = str(cfg.det_limit_type)
        if cfg.det_db_thresh is not None:
            kwargs["det_db_thresh"] = float(cfg.det_db_thresh)
        if cfg.det_db_box_thresh is not None:
            kwargs["det_db_box_thresh"] = float(cfg.det_db_box_thresh)
        if cfg.det_db_unclip_ratio is not None:
            kwargs["det_db_unclip_ratio"] = float(cfg.det_db_unclip_ratio)

        self.ocr = PaddleOCR(**kwargs)

    def detect_and_recognize(
        self,
        image_path: str,
        y_tol: int = 25,
        min_score: float = 0.0,
        san_like_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        返回排序后的文本项：
        [{"text": "...", "bbox": [x1,y1,x2,y2]}, ...]
        """
        pil_rgb = load_rgb_image(image_path)
        img_bgr = cv2.cvtColor(np.asarray(pil_rgb), cv2.COLOR_RGB2BGR)
        result = self.ocr.ocr(img_bgr, cls=False)
        # PaddleOCR 返回结构通常为：[[[poly], (text, score)], ...]
        if not result or not result[0]:
            return []

        items: List[Dict[str, Any]] = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            poly = line[0]
            txt, score = line[1]
            if txt is None:
                continue
            bbox = _xyxy_from_poly(poly)
            norm = normalize_ocr_text(txt)
            sc = float(score)
            if sc < float(min_score):
                continue
            if san_like_only:
                if looks_like_round_index(norm):
                    continue
                if not is_san_like(norm):
                    continue
            items.append({"text": norm, "bbox": bbox, "score": sc})

        items = sort_by_bbox(items, y_tol=y_tol)
        return items


def _is_rec_load_failure(exc: BaseException) -> bool:
    msg = str(exc)
    if "load_combine" in msg or "Not allowed to load partial" in msg or "partial data" in msg:
        return True
    # 与 load_combine 相关的 load_op 提示
    return "load_op" in msg and "Unavailable" in msg


def default_predictor_with_notice(extra_cfg: Optional[OcrPredictorConfig] = None) -> Tuple[OcrPredictor, Optional[str]]:
    """
    默认加载：
    - 检测：若 det_chess_infer 可用则微调检测，否则 Paddle 默认英文检测。
    - 识别：若 rec_chess_infer 可用则微调识别，否则默认英文识别。
    - 识别权重损坏时回退默认识别并保留微调检测（若可用）。

    返回 (预测器, 可选提示)。
    """
    root = Path(__file__).resolve().parents[1]

    dict_path = root / "data" / "dict_chess.txt"
    rec_dict = str(dict_path) if dict_path.exists() else None

    rec_infer_dir = root / "models" / "finetuned" / "rec_chess_infer"

    det_infer_dir = root / "models" / "finetuned" / "det_chess_infer"

    notices: List[str] = []

    det_dir: Optional[str] = None
    if _infer_model_dir_ready(det_infer_dir):
        det_dir = _ensure_paddle_readable_rec_dir(det_infer_dir)
    elif det_infer_dir.exists() and any(det_infer_dir.iterdir()):
        notices.append(
            "微调检测模型不完整或异常（需 inference.pdmodel + inference.pdiparams），已改用 Paddle 默认英文检测。"
            "训练导出：python scripts/4_export_inference_det.py"
        )

    rec_dir: Optional[str] = None
    if _infer_model_dir_ready(rec_infer_dir):
        rec_dir = _ensure_paddle_readable_rec_dir(rec_infer_dir)
    elif rec_infer_dir.exists() and any(rec_infer_dir.iterdir()):
        notices.append(
            "微调识别模型不完整或异常，已改用 PaddleOCR 默认英文识别。"
            "导出：python scripts/3_export_inference.py"
        )

    merged = OcrPredictorConfig(
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
        rec_char_dict_path=rec_dict if rec_dir else None,
        use_angle_cls=False,
        lang="en",
    )
    if extra_cfg is not None:
        merged = OcrPredictorConfig(
            **{**merged.__dict__, **{k: v for k, v in extra_cfg.__dict__.items() if v is not None}}
        )

    notice_txt = "\n".join(notices) if notices else None

    try:
        return OcrPredictor(merged), notice_txt
    except Exception as e:
        if not _is_rec_load_failure(e):
            raise
        _clear_rec_ascii_cache(rec_infer_dir)
        fallback = OcrPredictorConfig(
            det_model_dir=merged.det_model_dir,
            rec_model_dir=None,
            rec_char_dict_path=None,
            use_angle_cls=False,
            lang="en",
        )
        if extra_cfg is not None:
            fallback = OcrPredictorConfig(
                **{**fallback.__dict__, **{k: v for k, v in extra_cfg.__dict__.items() if v is not None}}
            )
        fix_notice = (
            (notice_txt + "\n") if notice_txt else ""
        ) + (
            "微调识别模型加载失败，已退回默认英文识别（保留微调检测若可用）。"
            "请重新导出：python scripts\\3_export_inference.py"
        )
        return OcrPredictor(fallback), fix_notice


def default_predictor() -> OcrPredictor:
    pred, _ = default_predictor_with_notice()
    return pred

