"""
步骤 1b：检测（文本框）训练数据准备

将每张棋谱大图上的 LabelMe / 简化 JSON 标注转为 PaddleOCR DB 检测训练所需的标签格式：
每行：相对路径（相对项目根）\\t JSON 列表
JSON 每项：{"transcription": "...", "points": [[x,y], ...]} （四点框，轴对齐）

与识别数据使用同一套图片与 train/lable 下的 JSON（由 find_input_dirs 决定）。

输出：
- data/det_train_label.txt
- data/det_eval_label.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from utils import (
    find_input_dirs,
    imread_bgr_unicode,
    list_images,
    parse_annotation_items,
    project_root,
    read_json,
    write_lines,
)


def _bbox_to_quad(x1: int, y1: int, x2: int, y2: int) -> List[List[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _line_for_image(
    root: Path, img_path: Path, ann_path: Path
) -> str | None:
    bgr = imread_bgr_unicode(img_path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    ann = read_json(ann_path)
    items = parse_annotation_items(ann, image_w=w, image_h=h)
    records: List[dict[str, Any]] = []
    for it in items:
        text = str(it.text).strip()
        if not text:
            continue
        x1, y1, x2, y2 = it.bbox
        records.append(
            {
                "transcription": text,
                "points": _bbox_to_quad(x1, y1, x2, y2),
            }
        )
    if not records:
        return None
    rel = img_path.resolve().relative_to(root.resolve()).as_posix()
    return f"{rel}\t{json.dumps(records, ensure_ascii=False)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    root = project_root()
    img_dir, ann_dir = find_input_dirs(root)
    images = list_images(img_dir)
    if not images:
        raise SystemExit(
            f"未找到图片。请把棋谱图放在 {img_dir}，并在 {ann_dir} 提供同名 .json 标注。"
        )

    lines: List[str] = []
    for img_path in tqdm(images, desc="Build det labels"):
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.json"
        if not ann_path.exists():
            continue
        line = _line_for_image(root, img_path, ann_path)
        if line:
            lines.append(line)

    if not lines:
        raise SystemExit("没有生成任何检测标签。请检查 JSON 与图片是否成对、box 是否有效。")

    rng = random.Random(int(args.seed))
    rng.shuffle(lines)
    n_train = int(len(lines) * float(args.train_ratio))
    train_lines = lines[:n_train]
    eval_lines = lines[n_train:]

    if not eval_lines and len(train_lines) > 1:
        # 至少留 1 条做 eval
        eval_lines = [train_lines.pop()]

    out_train = root / "data" / "det_train_label.txt"
    out_eval = root / "data" / "det_eval_label.txt"
    write_lines(out_train, train_lines)
    write_lines(out_eval, eval_lines)

    print("Done (text detection / DB).")
    print(f"- {out_train} ({len(train_lines)} images)")
    print(f"- {out_eval}  ({len(eval_lines)} images)")


if __name__ == "__main__":
    main()
