"""
步骤 2b：微调文本检测模型（英文 PP-OCRv3 / DB 检测分支）

PaddleOCR 英语推理包使用的检测结构与官方 en_PP-OCRv3_det 一致（MobileNetV3 + DB）。
预训练权重使用官方蒸馏训练包中的 student：
  en_PP-OCRv3_det_distill_train.tar

流程：
1) python scripts/1_prepare_det_data.py
2) python scripts/2_train_det.py

导出推理模型：
  python scripts/4_export_inference_det.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from utils import ensure_dir, guess_paddleocr_repo_root, project_root, download_and_extract_zip


PADDLEOCR_ZIP_URLS = [
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release/2.7/PaddleOCR-release-2.7.zip",
    "https://gitcode.net/paddlepaddle/PaddleOCR/-/archive/release%2F2.7/PaddleOCR-release-2.7.zip",
    "https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/release/2.7.zip",
]

EN_DET_DISTILL_TAR_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar"
)


def resolve_det_train_config(cfg_path: Path, crop_size: int) -> Path:
    """
    Paddle 的 merge_config 无法用 Train.dataset.transforms.3.xxx 覆盖列表项（会把 '3' 当 dict 键）。
    若 --crop_size 与 yml 中 EastRandomCropData.size 不一致，则写入缓存副本再传给 train.py。
    """
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("需要 PyYAML（pip install pyyaml）才能按 crop_size 生成训练配置。") from e

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    transforms = cfg.get("Train", {}).get("dataset", {}).get("transforms")
    if not isinstance(transforms, list):
        raise SystemExit("训练配置缺少 Train.dataset.transforms 列表")

    found = False
    for block in transforms:
        if isinstance(block, dict) and "EastRandomCropData" in block:
            found = True
            sub = block["EastRandomCropData"]
            cur = sub.get("size")
            if (
                isinstance(cur, (list, tuple))
                and len(cur) == 2
                and int(cur[0]) == int(crop_size)
                and int(cur[1]) == int(crop_size)
            ):
                return cfg_path
            sub["size"] = [int(crop_size), int(crop_size)]
            break
    if not found:
        raise SystemExit("配置中未找到 EastRandomCropData，无法应用 --crop_size")

    local = os.environ.get("LOCALAPPDATA", "")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    cache = base / "project-chess-ocr-cache" / "det_train_cfg"
    ensure_dir(cache)
    out = cache / f"{cfg_path.stem}_crop{crop_size}.yml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return out


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
            f"你可以手动下载 release/2.7 zip 放到 {cache_dir} 后重试。\n最后错误：{last_err}"
        )
    repo_root = guess_paddleocr_repo_root(extracted)
    if not (repo_root / "tools" / "train.py").exists():
        raise SystemExit(f"未找到 PaddleOCR tools/train.py：{repo_root}")
    return repo_root


def _locate_student_pdparams(search_root: Path) -> Path | None:
    direct = search_root / "student.pdparams"
    if direct.is_file():
        return direct
    for p in search_root.rglob("student.pdparams"):
        return p
    return None


def ensure_pretrained_det_student(root: Path) -> Path:
    """
    下载并解压 en_PP-OCRv3_det_distill_train，确保存在
    models/pretrained/en_PP-OCRv3_det_pretrained/student.pdparams
    """
    import urllib.request

    cache_dir = root / "models" / "pretrained"
    ensure_dir(cache_dir)

    out_dir = cache_dir / "en_PP-OCRv3_det_pretrained"
    expected = out_dir / "student.pdparams"

    if expected.is_file() and expected.stat().st_size > 4096:
        return expected

    tar_path = cache_dir / "en_PP-OCRv3_det_distill_train.tar"
    if not tar_path.exists():
        print(f"Downloading {EN_DET_DISTILL_TAR_URL}")
        req = urllib.request.Request(EN_DET_DISTILL_TAR_URL, headers={"User-Agent": "project-chess-ocr/1.0"})
        with urllib.request.urlopen(req) as resp, tar_path.open("wb") as f:
            f.write(resp.read())

    ensure_dir(out_dir)
    print(f"Extracting det pretrained tar -> {out_dir}")
    try:
        subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(out_dir)])
    except Exception as e:
        raise SystemExit(
            f"解压失败，请确认系统有 tar，或手动解压 {tar_path} 到 {out_dir}。\n{e}"
        )

    src = _locate_student_pdparams(out_dir)
    if src is None or not src.is_file():
        raise SystemExit(f"未在解压目录中找到 student.pdparams：{out_dir}")

    if not expected.exists() or expected.resolve() != src.resolve():
        shutil.copy2(src, expected)

    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/det_en_ppocrv3_student_chess.yml",
        help="检测训练配置 yml",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="每步批量；CPU 上宁可 2~4 减轻单步耗时（总 iter 会变多，但单步更快、更省内存）",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=736,
        choices=[512, 640, 736, 832, 960],
        help="EastRandomCrop 边长；棋谱字行较密时 736/960 往往比 640 更利于召回（CPU 会更慢）",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="从第 0 步之后，每隔多少 iter 跑一次验证（拉大可减少验证占用时间）",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=60,
        help="微调 epoch；漏框仍多时可试 80～120，并结合日志 best metric 是否还在提升",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=5,
        help="每隔多少 step 打印一行训练统计（loss、eta 等），便于看进度",
    )
    parser.add_argument("--paddleocr_repo", type=str, default="", help="本地 PaddleOCR 仓库路径（可选）")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="断点续训：权重前缀路径（需存在 xxx.pdparams），如 models/finetuned/det_chess/latest",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="仅加载权重热启动（新优化器），如 models/finetuned/det_chess/best_accuracy；与 --resume 互斥",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="覆盖 Optimizer.lr.learning_rate；断点续训时常用 5e-4 或 3e-4",
    )
    parser.add_argument(
        "--warmup_epoch",
        type=int,
        default=None,
        help="覆盖 Optimizer.lr.warmup_epoch（续训时可改小，如 0～1）",
    )
    parser.add_argument(
        "--post_thresh",
        type=float,
        default=0.22,
        help="验证用 PostProcess.thresh（越低越偏向召回）",
    )
    parser.add_argument(
        "--post_box_thresh",
        type=float,
        default=0.45,
        help="验证用 PostProcess.box_thresh（越低越偏向召回）",
    )
    parser.add_argument(
        "--post_unclip",
        type=float,
        default=1.65,
        help="验证用 PostProcess.unclip_ratio（略大可缓解框偏小，过大易粘连）",
    )
    args = parser.parse_args()

    root = project_root()

    train_label = root / "data" / "det_train_label.txt"
    eval_label = root / "data" / "det_eval_label.txt"
    if not train_label.exists() or not eval_label.exists():
        raise SystemExit("缺少 data/det_train_label.txt 或 data/det_eval_label.txt，请先运行 scripts/1_prepare_det_data.py")

    if args.paddleocr_repo:
        repo_root = Path(args.paddleocr_repo).resolve()
        repo_root = guess_paddleocr_repo_root(repo_root)
        if not (repo_root / "tools" / "train.py").exists():
            raise SystemExit(f"--paddleocr_repo 无效：{repo_root}")
    else:
        repo_root = ensure_paddleocr_repo(root)

    student_pp = ensure_pretrained_det_student(root)
    pretrained_prefix = student_pp.with_suffix("")  # strip .pdparams for Paddle load

    if args.resume and args.weights:
        raise SystemExit("--resume 与 --weights 不能同时使用")

    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"找不到配置文件：{cfg_path}")

    cs = int(args.crop_size)
    train_cfg = resolve_det_train_config(cfg_path, cs)
    if train_cfg.resolve() != cfg_path.resolve():
        print(f"已按 crop={cs} 写入临时配置（Paddle 无法用 -o 覆盖 transforms 列表项）：\n  {train_cfg}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    train_py = repo_root / "tools" / "train.py"

    ev = max(10, int(args.eval_every))

    cmd = [
        sys.executable,
        str(train_py),
        "-c",
        str(train_cfg),
        "-o",
        f"Global.epoch_num={int(args.epoch)}",
        f"Train.loader.batch_size_per_card={int(args.batch_size)}",
        f"Eval.loader.batch_size_per_card=1",
        f"Global.print_batch_step={max(1, int(args.print_step))}",
        f"Global.eval_batch_step=[0,{ev}]",
        "Global.use_gpu=False",
        "Global.distributed=False",
        "Train.loader.num_workers=0",
        "Eval.loader.num_workers=0",
        "Train.loader.use_shared_memory=False",
        f"PostProcess.thresh={float(args.post_thresh)}",
        f"PostProcess.box_thresh={float(args.post_box_thresh)}",
        f"PostProcess.unclip_ratio={float(args.post_unclip)}",
    ]

    if args.resume:
        ck = Path(args.resume)
        ck = ck.resolve() if ck.is_absolute() else (root / ck).resolve()
        ck_str = ck.as_posix()
        if not Path(ck_str + ".pdparams").is_file():
            raise SystemExit(f"未找到断点权重文件：{ck_str}.pdparams")
        cmd.append(f"Global.checkpoints={ck_str}")
    elif args.weights:
        wp = Path(args.weights)
        wp = wp.resolve() if wp.is_absolute() else (root / wp).resolve()
        wp_str = wp.as_posix()
        exp = Path(wp_str + ".pdparams")
        if not exp.is_file():
            raise SystemExit(f"未找到热启动权重文件：{exp}")
        cmd.append(f"Global.pretrained_model={wp_str}")
    else:
        cmd.append(f"Global.pretrained_model={pretrained_prefix.as_posix()}")

    if args.lr is not None:
        cmd.append(f"Optimizer.lr.learning_rate={float(args.lr)}")
    if args.warmup_epoch is not None:
        cmd.append(f"Optimizer.lr.warmup_epoch={int(args.warmup_epoch)}")

    print("--- 检测模型微调 ---")
    print(
        "进度说明：每行日志含 epoch、global_step、loss、eta（剩余时间粗算）。\n"
        "ips≈每秒处理样本数；avg_batch_cost 为一步（整批）耗时。CPU+检测 本身很慢属正常。\n"
        f"当前：batch_size={int(args.batch_size)}, crop={cs}x{cs}, eval 每 {ev} iter。"
        " 加速试跑可加：更小 --batch_size 2、--crop_size 512、或装 GPU 后改用 GPU 版训练命令。\n"
        f"更密日志：--print_step 1 ；当前 print_step={max(1, int(args.print_step))}。\n"
        "漏框多：可加大 --crop_size、加长 --epoch；验证解码更偏召回可调 --post_thresh/--post_box_thresh。\n"
        "续训：--resume models/finetuned/det_chess/latest ；或 --weights models/finetuned/det_chess/best_accuracy。\n"
    )
    print("Running:", " ".join(cmd))

    log_dir = root / "models" / "finetuned" / "det_chess"
    ensure_dir(log_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_det_{ts}.log"
    print(f"Log: {log_path}")

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
    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"训练退出码 {ret}，见日志 {log_path}")

    print("检测模型训练结束。权重目录：", root / "models" / "finetuned" / "det_chess")


if __name__ == "__main__":
    main()
