"""
步骤 3：导出推理模型

使用 PaddleOCR 官方仓库的 tools/export_model.py，将训练得到的 best_accuracy
导出为推理格式（inference model），以便在 Streamlit App 中通过 PaddleOCR 直接加载。
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
    "https://ghproxy.com/https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
    "https://hub.njuu.cf/https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
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
        raise SystemExit(
            "下载 PaddleOCR 仓库失败。\n"
            f"你可以手动下载 release/2.7 zip 放到 {cache_dir}，命名为 PaddleOCR-release-2.7.zip 后重试。\n"
            f"最后错误：{last_err}"
        )
    repo_root = guess_paddleocr_repo_root(extracted)
    if not (repo_root / "tools" / "export_model.py").exists():
        raise SystemExit(f"未找到 PaddleOCR tools/export_model.py，解压目录为：{repo_root}")
    return repo_root


def _export_path_needs_ascii_staging(out_dir: Path) -> bool:
    """
    Windows 下若 Global.save_inference_dir 含非 ASCII（如中文目录名），
    paddle.jit.save 可能只写出 inference.pdmodel / .info，而漏写体积较大的 inference.pdiparams。
    此时先导出到 %LOCALAPPDATA% 下的纯 ASCII 暂存目录，再复制到目标路径。
    """
    try:
        str(out_dir.resolve()).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _ascii_export_staging_dir(out_dir: Path) -> Path:
    key = hashlib.sha1(str(out_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    local = os.environ.get("LOCALAPPDATA", "")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    staging = base / "project-chess-ocr-cache" / "export_staging" / key
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/rec_mv3_none_bilstm_ctc_chess.yml", help="训练配置 yml")
    parser.add_argument("--best_model_dir", type=str, default="models/finetuned/rec_chess", help="训练输出目录")
    parser.add_argument("--output_dir", type=str, default="models/finetuned/rec_chess_infer", help="导出推理模型目录")
    parser.add_argument(
        "--paddleocr_repo",
        type=str,
        default="",
        help="可选：本地 PaddleOCR 仓库路径（若你手动 git clone/下载解压了仓库，可用此参数跳过在线下载）",
    )
    args = parser.parse_args()

    root = project_root()
    if args.paddleocr_repo:
        repo_root = Path(args.paddleocr_repo).resolve()
        repo_root = guess_paddleocr_repo_root(repo_root)
        if not (repo_root / "tools" / "export_model.py").exists():
            raise SystemExit(f"--paddleocr_repo 路径下未找到 tools/export_model.py：{repo_root}")
    else:
        repo_root = ensure_paddleocr_repo(root)

    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"找不到配置文件：{cfg_path}")

    best_dir = (root / args.best_model_dir).resolve()
    if not best_dir.exists():
        raise SystemExit(f"找不到训练目录：{best_dir}（请先运行 scripts/2_train_rec.py）")

    out_dir = (root / args.output_dir).resolve()
    ensure_dir(out_dir)

    save_inference_dir = out_dir
    copy_to_out: Path | None = None
    if os.name == "nt" and _export_path_needs_ascii_staging(out_dir):
        save_inference_dir = _ascii_export_staging_dir(out_dir)
        copy_to_out = out_dir
        print(
            "注意：推理导出目标路径含非 ASCII 字符；为避免漏写 inference.pdiparams，"
            f"将先导出到：\n  {save_inference_dir}\n再复制到：\n  {out_dir}\n",
            flush=True,
        )

    export_py = repo_root / "tools" / "export_model.py"

    # PaddleOCR export_model.py 常用参数：
    # -c config
    # -o Global.checkpoints=... Global.save_inference_dir=...
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    # 默认导出 best_accuracy
    ckpt = best_dir / "best_accuracy"
    if not (ckpt.with_suffix(".pdparams")).exists():
        # 有些版本保存为 best_accuracy.pdparams / best_accuracy.pdopt 等
        # 这里只检查 pdparams 是否存在
        raise SystemExit(
            f"未找到 {ckpt}.pdparams。请确认训练是否完成，或检查 {best_dir} 下的权重文件名。"
        )

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

    print("Running export command:")
    print(" ".join(cmd))

    subprocess.check_call(cmd, cwd=str(root), env=env)

    if copy_to_out is not None:
        for name in ("inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"):
            src = save_inference_dir / name
            if src.is_file():
                shutil.copy2(src, copy_to_out / name)
        pdip = copy_to_out / "inference.pdiparams"
        if not pdip.is_file() or pdip.stat().st_size < 1024:
            raise SystemExit(
                f"复制推理权重失败或 inference.pdiparams 异常小。请手动从 {save_inference_dir} 复制到 {copy_to_out}"
            )

    print("Export finished.")
    print(f"Infer model dir: {out_dir}")


if __name__ == "__main__":
    main()

