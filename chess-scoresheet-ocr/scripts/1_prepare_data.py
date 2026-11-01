"""
步骤 1：数据准备脚本

目标：
- 读取原始棋谱照片与标注 JSON
- 按 bbox 裁剪出每一步走法的小图
- 做轻量图像预处理 + 数据增强（可控）
- 生成 PaddleOCR 识别训练所需的 train.txt / eval.txt（图片路径\t文本）
- 生成自定义字典 dict_chess.txt

兼容两种标注格式：
1) 简化格式（需求中假设的）：
   [{"text":"e4","bbox":[x1,y1,x2,y2]}, ...]
2) LabelMe（你现有数据）：
   {"shapes":[{"label":"e4","points":[[x1,y1],[x2,y2]],"shape_type":"rectangle"}, ...]}

目录约定：
- data/raw/ 与 data/raw_annotations/ 是推荐放置位置
- 若 data/raw 为空且存在 train/image + train/lable，则自动使用它们作为输入（兼容你当前数据结构）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

from utils import (
    OcrItem,
    ensure_dir,
    find_input_dirs,
    imread_bgr_unicode,
    imwrite_jpg_unicode,
    list_images,
    parse_annotation_items,
    preprocess_crop_for_rec,
    augment_crop,
    project_root,
    read_json,
    sort_items_reading_order,
    write_lines,
)


def build_chess_dict(extra_samples: List[str] | None = None) -> List[str]:
    """
    生成自定义词典（字符级）。

    PaddleOCR 的 CTC/Attention 字典通常是一行一个字符。
    因此这里放“字符集合”最关键：A-Z, a-z, 0-9, 以及棋谱常用符号。
    """
    chars = set()

    # 棋子字母（SAN 通常大写，但手写可能出现小写）
    for c in "KQRBNP":
        chars.add(c)
        chars.add(c.lower())

    # 坐标与数字
    for c in "abcdefgh":
        chars.add(c)
    for c in "12345678":
        chars.add(c)
    for c in "0-9":
        chars.add(c)
    for c in "0123456789":
        chars.add(c)

    # 常见符号
    for c in "+#=x-()":
        chars.add(c)
    # 棋谱里也会出现 O（王车易位）与空格、点号（e.p.）
    for c in "Oo":
        chars.add(c)
    chars.add(".")
    chars.add(" ")

    # 额外样例文本：可把其中出现过的字符也加入（增强覆盖）
    if extra_samples:
        for s in extra_samples:
            for ch in s:
                chars.add(ch)

    # PaddleOCR 通常不希望字典里有空行
    out = sorted(chars)
    # 去掉可能的空字符
    out = [c for c in out if c != ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例（其余为验证集）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于划分 train/eval 与增强）")
    parser.add_argument("--augment_factor", type=int, default=1, help="每个裁剪样本额外生成多少个增强版本（0 表示不增强）")
    parser.add_argument("--y_tol", type=int, default=25, help="排序时行合并阈值（像素）")
    args = parser.parse_args()

    root = project_root()
    img_dir, ann_dir = find_input_dirs(root)

    prepared_dir = root / "data" / "prepared"
    ensure_dir(prepared_dir)

    images = list_images(img_dir)
    if not images:
        raise SystemExit(
            f"未找到图片。请将图片放入 {root / 'data' / 'raw'}（或当前已有的 {root / 'train' / 'image'}）。"
        )

    rng = random.Random(args.seed)

    # 收集所有 label 文本，用于生成字典覆盖
    all_texts: List[str] = []
    pairs: List[Tuple[str, str]] = []  # (img_rel_path, text)

    for img_path in tqdm(images, desc="Preparing crops"):
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.json"
        if not ann_path.exists():
            # 有些图片可能没标注，直接跳过
            continue

        # 兼容中文路径
        bgr = imread_bgr_unicode(img_path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        ann = read_json(ann_path)
        items = parse_annotation_items(ann, image_w=w, image_h=h)
        items = sort_items_reading_order(items, y_tol=args.y_tol)

        for idx, it in enumerate(items):
            text = str(it.text).strip()
            if not text:
                continue

            x1, y1, x2, y2 = it.bbox
            crop = bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            crop = preprocess_crop_for_rec(crop)

            # 保存原始裁剪
            out_name = f"{stem}_{idx:04d}.jpg"
            out_path = prepared_dir / out_name
            if not imwrite_jpg_unicode(out_path, crop):
                continue

            rel_path = out_path.relative_to(root).as_posix()
            pairs.append((rel_path, text))
            all_texts.append(text)

            # 数据增强：生成若干增强版本
            for k in range(max(0, int(args.augment_factor))):
                aug = augment_crop(crop, rng=rng)
                aug_name = f"{stem}_{idx:04d}_aug{k:02d}.jpg"
                aug_path = prepared_dir / aug_name
                if not imwrite_jpg_unicode(aug_path, aug):
                    continue

                rel_aug = aug_path.relative_to(root).as_posix()
                pairs.append((rel_aug, text))

    if not pairs:
        raise SystemExit("未生成任何样本。请检查 JSON 标注与 bbox 是否正确。")

    # 随机打乱并划分
    rng.shuffle(pairs)
    n_train = int(len(pairs) * float(args.train_ratio))
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train:]

    train_txt = root / "data" / "train.txt"
    eval_txt = root / "data" / "eval.txt"
    write_lines(train_txt, [f"{p}\t{t}" for p, t in train_pairs])
    write_lines(eval_txt, [f"{p}\t{t}" for p, t in eval_pairs])

    # 生成字典
    dict_path = root / "data" / "dict_chess.txt"
    dict_chars = build_chess_dict(extra_samples=all_texts[:2000])
    write_lines(dict_path, dict_chars)

    print("Done.")
    print(f"- prepared_dir: {prepared_dir}")
    print(f"- train.txt: {train_txt} ({len(train_pairs)} lines)")
    print(f"- eval.txt:  {eval_txt} ({len(eval_pairs)} lines)")
    print(f"- dict:      {dict_path} ({len(dict_chars)} chars)")


if __name__ == "__main__":
    main()

