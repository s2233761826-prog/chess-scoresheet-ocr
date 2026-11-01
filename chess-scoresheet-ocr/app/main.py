"""
Streamlit 客户端主程序

交互流程：
1) 上传图片
2) 点击“开始识别”：运行 det+rec，得到走法列表（按 bbox 排序）
3) 点击“校验与校正”：用 python-chess 从头校验；遇到第一处非法走法：
   - 给出 1~3 个候选 SAN（基于 Levenshtein）
   - 用户选择候选或手动输入
   - 确认后继续校验下一步
4) 校验全部通过后，生成 PGN，支持下载

备注：
- Streamlit 是 rerun 模式；因此我们把识别结果、用户修正记录、校验进度放在 session_state。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from chess_validator import ChessCorrector
from ocr_predictor import OcrPredictorConfig, default_predictor_with_notice, load_rgb_image
from pgn_builder import moves_to_pgn


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "corrections_log"


def init_state() -> None:
    st.session_state.setdefault("image_path", None)
    st.session_state.setdefault("ocr_items_raw", [])
    st.session_state.setdefault("ocr_items", [])  # [{"text","bbox","score"}...]
    st.session_state.setdefault("move_texts", [])  # ["e4","e5",...]
    st.session_state.setdefault("chosen_moves", {})  # {index: "SAN"}
    st.session_state.setdefault("corrected_moves", [])  # 已确认的 SAN 列表（连续前缀）
    st.session_state.setdefault("issues", [])  # 当前需要处理的 issue（最多一个）
    st.session_state.setdefault("pgn", "")


def save_uploaded_file(uploaded) -> str:
    ensure_dir = ROOT / "data" / "_uploads"
    ensure_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    out = ensure_dir / f"upload_{ts}{suffix}"
    out.write_bytes(uploaded.getbuffer())
    return str(out)


def log_correction(image_path: str, move_texts: List[str], chosen_moves: Dict[int, str], corrected_moves: List[str]) -> None:
    """
    保存用户修正日志，供后续继续训练/分析。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = LOG_DIR / f"correction_{ts}.json"
    payload: Dict[str, Any] = {
        "time": ts,
        "image_path": image_path,
        "ocr_moves": move_texts,
        "chosen_moves": {str(k): v for k, v in chosen_moves.items()},
        "corrected_moves": corrected_moves,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _annotate_like_labelme(path: str, items: List[Dict[str, Any]]) -> Image.Image:
    """
    用类似 LabelMe 的风格把 bbox + label 画到原图上：
    - 矩形框
    - 左上角 label（带底色）
    """
    img = load_rgb_image(path)
    draw = ImageDraw.Draw(img)
    # 尽量使用系统字体；失败则 fallback
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    for it in items:
        bbox = it.get("bbox")
        text = it.get("text") or ""
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        # box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        if text:
            label = str(text)
            # text bg
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 4
            bx1, by1 = x1, max(0, y1 - th - pad * 2)
            bx2, by2 = x1 + tw + pad * 2, by1 + th + pad * 2
            draw.rectangle([bx1, by1, bx2, by2], fill=(255, 0, 0))
            draw.text((bx1 + pad, by1 + pad), label, fill=(255, 255, 255), font=font)

    return img


def main() -> None:
    st.set_page_config(page_title="国际象棋手写棋谱 OCR 与校正系统", layout="wide")
    init_state()

    st.title("国际象棋手写棋谱 OCR 与校正系统")

    with st.sidebar:
        st.header("上传棋谱照片")
        uploaded = st.file_uploader("选择一张棋谱照片（jpg/png）", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            path = save_uploaded_file(uploaded)
            st.session_state["image_path"] = path

        st.markdown("---")
        st.subheader("识别参数（减少噪声）")
        min_score = st.slider("最低置信度（rec score）", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
        san_like_only = st.checkbox("只保留像棋谱走法的文本（推荐）", value=True)
        show_all_boxes = st.checkbox("右侧显示全部检测框（调试）", value=False)
        with st.expander("检测框参数（DB，默认与 PaddleOCR 一致）", expanded=False):
            det_limit_side_len = st.slider("det_limit_side_len（输入边长限制）", min_value=320.0, max_value=1600.0, value=960.0, step=16.0)
            det_limit_type = st.selectbox("det_limit_type", options=["max", "min"], index=0)
            det_db_thresh = st.slider("det_db_thresh（二值化阈值，越大框越少）", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
            det_db_box_thresh = st.slider("det_db_box_thresh（框分数阈值，越大框越少）", min_value=0.10, max_value=0.95, value=0.60, step=0.05)
            det_db_unclip_ratio = st.slider("det_db_unclip_ratio（框扩张，过大易粘连）", min_value=1.0, max_value=2.5, value=1.5, step=0.05)
        st.caption("提示：首次运行 PaddleOCR 可能会自动下载模型，请耐心等待。")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("原图（上传）")
        if st.session_state["image_path"]:
            img_path = str(st.session_state["image_path"])
            try:
                img = load_rgb_image(img_path)
                st.image(img, caption=f"原图：{Path(img_path).name}（{img.width}×{img.height}）", use_column_width=True)
            except Exception as e:
                p = Path(img_path)
                st.error(f"原图读取失败：{e}")
                st.code(f"path={img_path}\nexists={p.exists()}\nsize={p.stat().st_size if p.exists() else 'NA'}")
        else:
            st.info("请先在左侧上传棋谱照片。")

        st.markdown("---")
        st.subheader("识别列表（文本 + bbox）")

        if st.button("开始识别", type="primary", disabled=st.session_state["image_path"] is None):
            det_extra = OcrPredictorConfig(
                det_limit_side_len=float(det_limit_side_len),
                det_limit_type=str(det_limit_type),
                det_db_thresh=float(det_db_thresh),
                det_db_box_thresh=float(det_db_box_thresh),
                det_db_unclip_ratio=float(det_db_unclip_ratio),
            )
            predictor, ocr_notice = default_predictor_with_notice(det_extra)
            if ocr_notice:
                st.warning(ocr_notice)
            raw_items = predictor.detect_and_recognize(
                st.session_state["image_path"],
                min_score=float(min_score),
                san_like_only=bool(san_like_only),
            )
            items = list(raw_items)
            st.session_state["ocr_items_raw"] = raw_items
            st.session_state["ocr_items"] = items
            st.session_state["move_texts"] = [it["text"] for it in items if it.get("text")]
            st.session_state["chosen_moves"] = {}
            st.session_state["corrected_moves"] = []
            st.session_state["issues"] = []
            st.session_state["pgn"] = ""

        if st.session_state["ocr_items"]:
            with st.expander("展开查看识别表格（已按 bbox 阅读顺序排序）", expanded=False):
                st.dataframe(
                    [
                        {"idx": i, "text": it["text"], "score": round(it.get("score", 0.0), 4), "bbox": it["bbox"]}
                        for i, it in enumerate(st.session_state["ocr_items"])
                    ],
                    use_container_width=True,
                )
        else:
            st.caption("提示：上传图片后点击“开始识别”。")

    with col2:
        st.subheader("推理结果标注图（LabelMe 风格）")
        ann_items = st.session_state["ocr_items_raw"] if show_all_boxes else st.session_state["ocr_items"]
        if st.session_state["image_path"] and ann_items:
            try:
                ann = _annotate_like_labelme(st.session_state["image_path"], ann_items)
                title = "推理标注图（全部检测框）" if show_all_boxes else "推理标注图（筛选后的走法框）"
                st.image(ann, caption=title, use_column_width=True)
            except Exception as e:
                st.error(f"渲染标注图失败：{e}")
        else:
            st.caption("完成“开始识别”后，这里会显示标注后的图片。")

        st.markdown("---")
        st.subheader("校验与校正（按国际象棋规则）")

        if st.button("校验与校正", disabled=not st.session_state["move_texts"]):
            corrector = ChessCorrector(topk=3)
            corrected, issues = corrector.validate_and_correct_interactive(
                st.session_state["move_texts"],
                st.session_state["chosen_moves"],
            )
            st.session_state["corrected_moves"] = corrected
            st.session_state["issues"] = [iss.__dict__ for iss in issues]

            if not issues:
                st.success("全部走法均合法（或已被修正）。")
                pgn = moves_to_pgn(corrected, result="*")
                st.session_state["pgn"] = pgn
                if st.session_state["image_path"]:
                    log_correction(
                        st.session_state["image_path"],
                        st.session_state["move_texts"],
                        st.session_state["chosen_moves"],
                        corrected,
                    )

        # 展示当前进度
        if st.session_state["move_texts"]:
            st.write(f"已确认走法：{len(st.session_state['corrected_moves'])} / {len(st.session_state['move_texts'])}")

        # 若存在 issue，渲染修正表单
        if st.session_state["issues"]:
            issue = st.session_state["issues"][0]
            idx = int(issue["index"])
            original = issue["original"]
            candidates = issue.get("candidates") or []

            st.warning(f"第 {idx} 步走法不合法或无法解析：`{original}`")

            with st.form(key="correction_form"):
                choice = st.radio(
                    "请选择候选走法（或使用下方手动输入）",
                    options=(candidates + ["(手动输入)"]) if candidates else ["(手动输入)"],
                    index=0,
                )
                manual = st.text_input("手动输入 SAN（例如：Nf3, O-O, exd5+）", value="")
                submitted = st.form_submit_button("确认该步并继续")

                if submitted:
                    if choice != "(手动输入)":
                        chosen = choice
                    else:
                        chosen = manual.strip()

                    st.session_state["chosen_moves"][idx] = chosen
                    # 触发下一次校验（Streamlit rerun 后用户点击“校验与校正”即可继续）
                    st.success(f"已记录第 {idx} 步修正为：{chosen}。请再次点击“校验与校正”继续。")

        # PGN 输出与下载
        if st.session_state.get("pgn"):
            st.subheader("PGN 预览")
            st.code(st.session_state["pgn"])
            st.caption("提示：右上角的复制按钮可直接复制 PGN。")

            st.download_button(
                label="下载 PGN 文件",
                data=st.session_state["pgn"].encode("utf-8"),
                file_name="game.pgn",
                mime="application/x-chess-pgn",
            )


if __name__ == "__main__":
    main()

