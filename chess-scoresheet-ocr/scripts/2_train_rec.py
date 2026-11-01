"""
步骤 2：训练识别模型（微调 en_PP-OCRv4_rec）

由于 PaddleOCR 的训练/导出工具通常随官方仓库一起发布，本脚本会自动：
1) 下载 PaddleOCR 官方仓库 zip（缓存到 models/pretrained/）
2) 下载英文识别预训练权重（缓存到 models/pretrained/）
3) 使用 tools/train.py 启动训练（CPU 友好，自动 use_gpu=False）

你只需确保先执行：
- python scripts/1_prepare_data.py

然后运行：
- python scripts/2_train_rec.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from utils import (
    ensure_dir,
    guess_paddleocr_repo_root,
    project_root,
    download_and_extract_zip,
)


# PaddleOCR 官方仓库 zip。GitHub 在部分网络环境下可能不稳定，这里提供备用镜像前缀。
PADDLEOCR_ZIP_URLS = [
    # 首选：GitCode / AtomGit 镜像（国内更容易访问）
    # GitCode 仓库页：https://gitcode.net/paddlepaddle/PaddleOCR
    # 归档 zip（GitLab 风格）：/-/archive/<branch>/<repo>-<branch>.zip
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release/2.7/PaddleOCR-release-2.7.zip",
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release%2F2.7/PaddleOCR-release-2.7.zip",
    # 备用：GitHub（若可访问）
    "https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
    # 备用：GitHub 代理（部分网络可用）
    "https://ghproxy.com/https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
    "https://hub.njuu.cf/https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
]

# 识别预训练权重（训练用：.pdparams/.pdopt/.states）
# 该链接来自 PaddleOCR 官方模型发布（若后续变更，可在 README 中替换）
EN_REC_PRETRAINED_TAR_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar"


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
            "下载 PaddleOCR 仓库失败（网络连接被远端关闭）。\n"
            f"你可以手动下载 release/2.7 zip 放到 {cache_dir}，命名为 PaddleOCR-release-2.7.zip 后重试。\n"
            f"最后错误：{last_err}"
        )
    repo_root = guess_paddleocr_repo_root(extracted)
    if not (repo_root / "tools" / "train.py").exists():
        raise SystemExit(f"未找到 PaddleOCR tools/train.py，解压目录为：{repo_root}")
    return repo_root


def ensure_pretrained_rec(root: Path) -> Path:
    """
    下载并解压 en_PP-OCRv4_rec 训练权重。

    PaddleOCR 的 train 权重是 tar 包；这里用系统 tar（Windows 10+ 通常自带 bsdtar）。
    若你的系统没有 tar，可手动解压到该目录。
    """
    import urllib.request

    cache_dir = root / "models" / "pretrained"
    ensure_dir(cache_dir)

    tar_path = cache_dir / "en_PP-OCRv4_rec_train.tar"
    if not tar_path.exists():
        print(f"Downloading pretrained rec tar to {tar_path}")
        req = urllib.request.Request(EN_REC_PRETRAINED_TAR_URL, headers={"User-Agent": "project-chess-ocr/1.0"})
        with urllib.request.urlopen(req) as resp, tar_path.open("wb") as f:
            f.write(resp.read())

    out_dir = cache_dir / "en_PP-OCRv4_rec_pretrained"
    if out_dir.exists() and any(out_dir.iterdir()):
        # 兼容：有些 tar 会解压出一层子目录（如 en_PP-OCRv4_rec_train/）
        # 我们希望最终存在 out_dir/best_accuracy.pdparams 等文件，供配置中的
        # Global.pretrained_model=./models/pretrained/en_PP-OCRv4_rec_pretrained/best_accuracy 使用。
        _ensure_best_accuracy_at_root(out_dir)
        return out_dir

    ensure_dir(out_dir)
    print(f"Extracting pretrained rec to {out_dir}")

    # Windows PowerShell 下调用 tar 解压
    try:
        subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(out_dir)])
    except Exception as e:
        raise SystemExit(
            "解压预训练权重失败。请确认系统存在 tar 命令，或手动解压 "
            f"{tar_path} 到 {out_dir}。\n"
            f"原始错误：{e}"
        )

    _ensure_best_accuracy_at_root(out_dir)
    return out_dir


def _ensure_best_accuracy_at_root(out_dir: Path) -> None:
    """
    将 best_accuracy.* 放到 out_dir 根目录。

    官方 tar 常见结构：
      out_dir/en_PP-OCRv4_rec_train/best_accuracy.pdparams
    而我们的训练配置期望：
      out_dir/best_accuracy.pdparams
    """
    root_pdparams = out_dir / "best_accuracy.pdparams"
    if root_pdparams.exists():
        return

    # 常见解压子目录名
    candidates = [
        out_dir / "en_PP-OCRv4_rec_train",
        out_dir / "en_PP-OCRv4_rec_train" / "best_accuracy.pdparams",
    ]

    # 统一找一次子目录下的 best_accuracy.*（不依赖具体目录名）
    sub_pdparams = None
    for p in out_dir.rglob("best_accuracy.pdparams"):
        sub_pdparams = p
        break
    if sub_pdparams is None:
        return

    sub_dir = sub_pdparams.parent
    for ext in [".pdparams", ".pdopt", ".states"]:
        src = sub_dir / ("best_accuracy" + ext)
        dst = out_dir / ("best_accuracy" + ext)
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/rec_mv3_none_bilstm_ctc_chess.yml", help="训练配置 yml")
    parser.add_argument("--batch_size", type=int, default=64, help="CPU 建议 16~64")
    parser.add_argument("--epoch", type=int, default=30, help="微调 epoch 数")
    parser.add_argument(
        "--paddleocr_repo",
        type=str,
        default="",
        help="可选：本地 PaddleOCR 仓库路径（若你手动 git clone/下载解压了仓库，可用此参数跳过在线下载）",
    )
    args = parser.parse_args()

    root = project_root()

    # 基本文件检查（避免训练启动后才报错）
    train_txt = root / "data" / "train.txt"
    eval_txt = root / "data" / "eval.txt"
    dict_path = root / "data" / "dict_chess.txt"
    if not train_txt.exists() or not eval_txt.exists() or not dict_path.exists():
        raise SystemExit("缺少 data/train.txt / data/eval.txt / data/dict_chess.txt，请先运行 scripts/1_prepare_data.py")

    if args.paddleocr_repo:
        repo_root = Path(args.paddleocr_repo).resolve()
        # 兼容用户传入的是“解压外层目录”
        repo_root = guess_paddleocr_repo_root(repo_root)
        if not (repo_root / "tools" / "train.py").exists():
            raise SystemExit(f"--paddleocr_repo 路径下未找到 tools/train.py：{repo_root}")
    else:
        repo_root = ensure_paddleocr_repo(root)
    pretrained_dir = ensure_pretrained_rec(root)

    # 把配置里用到的 pretrained_model 路径对齐到我们缓存目录
    # 训练配置里写的是 ./models/pretrained/en_PP-OCRv4_rec_pretrained/best_accuracy
    # 解压后一般会包含 best_accuracy.pdparams 等文件，目录结构由官方 tar 决定
    # 这里不强行改名，而是在必要时做一个软兼容：若 best_accuracy 不存在，则提示用户检查。
    expected = root / "models" / "pretrained" / "en_PP-OCRv4_rec_pretrained"
    if not expected.exists():
        # 将解压目录同步到期望位置
        expected.parent.mkdir(parents=True, exist_ok=True)
        if pretrained_dir != expected:
            # Windows 下避免符号链接，直接复用同一路径名
            # 如果解压到 out_dir 已是 expected，则不会进入这里
            pass

    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"找不到配置文件：{cfg_path}")

    # 为了让 yml 中相对路径生效，训练工作目录切到项目根
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU（即使装了 CUDA）

    # PaddleOCR 训练脚本
    train_py = repo_root / "tools" / "train.py"

    cmd = [
        sys.executable,
        str(train_py),
        "-c",
        str(cfg_path),
        "-o",
        f"Global.epoch_num={int(args.epoch)}",
        f"Train.loader.batch_size_per_card={int(args.batch_size)}",
        f"Eval.loader.batch_size_per_card={int(args.batch_size)}",
        "Global.use_gpu=False",
    ]

    print("Running training command:")
    print(" ".join(cmd))

    # 实时输出训练进展，并写入日志文件，便于断点排查与回看
    log_dir = root / "models" / "finetuned" / "rec_chess"
    ensure_dir(log_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"
    print(f"Training log will be saved to: {log_path}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    last_output_t = time.time()
    heartbeat_sec = 60  # 如果 60 秒没新日志，打印一次心跳提示

    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            last_output_t = time.time()
            # 同时输出到控制台与日志文件
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

            # 控制台心跳：避免用户误以为“卡住”
            if time.time() - last_output_t > heartbeat_sec:
                msg = f"[heartbeat] training is running... see log: {log_path}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                f.write(msg)
                f.flush()
                last_output_t = time.time()

    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"训练进程退出码非 0：{ret}。请查看日志：{log_path}")

    print("Training finished.")
    print(f"Model dir: {root / 'models' / 'finetuned' / 'rec_chess'}")


if __name__ == "__main__":
    main()

