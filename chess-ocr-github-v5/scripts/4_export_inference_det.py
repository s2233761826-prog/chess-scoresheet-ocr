"""
步骤 4b：将微调后的检测模型导出为推理格式（inference.pdmodel + inference.pdiparams）

在训练完成后执行：
  python scripts/4_export_inference_det.py

若项目路径含中文，本脚本会与你识别导出一样，先写入 %LOCALAPPDATA% 再复制到
  models/finetuned/det_chess_infer
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils import ensure_dir, guess_paddleocr_repo_root, project_root, download_and_extract_zip

PADDLEOCR_ZIP_URLS = [
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release/2.7/PaddleOCR-release-2.7.zip",
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release%2F2.7/PaddleOCR-release-2.7.zip",
    "https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
]


def ensure_paddleocr_repo(root: Path) -> Path:
    cache_dir = root / "models" / "pretrained"
    last_err: Exception | None = None
    extracted = None
    for url in PADDLEOCR_ZIP_URLS:
        try:
            extracted = download_and_extract_zip(url, cache_dir, "PaddleOCR-release-2.7.zip")
            break
        except Exception as e:
            last_err = e
            extracted = None
    if extracted is None:
        raise SystemExit(f"下载 PaddleOCR 仓库失败：{last_err}")
    repo_root = guess_paddleocr_repo_root(extracted)
    if not (repo_root / "tools" / "export_model.py").exists():
        raise SystemExit(f"未找到 tools/export_model.py：{repo_root}")
    return repo_root


def _export_path_needs_ascii_staging(out_dir: Path) -> bool:
    try:
        str(out_dir.resolve()).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _ascii_export_staging_dir(out_dir: Path) -> Path:
    key = hashlib.sha1(str(out_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    local = os.environ.get("LOCALAPPDATA", "")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    staging = base / "project-chess-ocr-cache" / "export_det_staging" / key
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/det_en_ppocrv3_student_chess.yml")
    parser.add_argument("--best_model_dir", type=str, default="models/finetuned/det_chess")
    parser.add_argument("--output_dir", type=str, default="models/finetuned/det_chess_infer")
    parser.add_argument("--paddleocr_repo", type=str, default="")
    args = parser.parse_args()

    root = project_root()

    if args.paddleocr_repo:
        repo_root = Path(args.paddleocr_repo).resolve()
        repo_root = guess_paddleocr_repo_root(repo_root)
        if not (repo_root / "tools" / "export_model.py").exists():
            raise SystemExit(f"--paddleocr_repo 无效：{repo_root}")
    else:
        repo_root = ensure_paddleocr_repo(root)

    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"找不到配置：{cfg_path}")

    best_dir = (root / args.best_model_dir).resolve()
    ckpt = best_dir / "best_accuracy"
    if not ckpt.with_suffix(".pdparams").exists():
        raise SystemExit(f"未找到 {ckpt}.pdparams，请先完成检测训练 scripts/2_train_det.py")

    out_dir = (root / args.output_dir).resolve()
    ensure_dir(out_dir)

    save_inference_dir = out_dir
    copy_to_out: Path | None = None
    if os.name == "nt" and _export_path_needs_ascii_staging(out_dir):
        save_inference_dir = _ascii_export_staging_dir(out_dir)
        copy_to_out = out_dir
        print(
            "推理导出路径含非 ASCII；先导出到：\n"
            f"  {save_inference_dir}\n再复制到：\n  {out_dir}\n",
            flush=True,
        )

    export_py = repo_root / "tools" / "export_model.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        str(export_py),
        "-c",
        str(cfg_path),
        "-o",
        f"Global.checkpoints={ckpt.as_posix()}",
        f"Global.save_inference_dir={save_inference_dir.as_posix()}",
        "Global.use_gpu=False",
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root), env=env)

    if copy_to_out is not None:
        for name in ("inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"):
            src = save_inference_dir / name
            if src.is_file():
                shutil.copy2(src, copy_to_out / name)
        pdip = copy_to_out / "inference.pdiparams"
        if not pdip.is_file() or pdip.stat().st_size < 1024:
            raise SystemExit(f"复制 inference 失败，请手动从 {save_inference_dir} 复制到 {copy_to_out}")

    print("导出完成：", out_dir)


if __name__ == "__main__":
    main()
