"""
PGN 生成模块

输入：按顺序排列的 SAN 走法列表（例如 ["e4", "e5", "Nf3"]）
输出：带回合号的 PGN 字符串（含基础标签）。

注意：
- 本模块假设输入走法是“合法 SAN”（已经通过 python-chess 校验/修正）。
- 结果标签（1-0/0-1/1/2-1/2/*）默认 "*"
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_TAGS = {
    "Event": "?",
    "Site": "?",
    "Date": datetime.now().strftime("%Y.%m.%d"),
    "Round": "?",
    "White": "?",
    "Black": "?",
    "Result": "*",
}


def moves_to_pgn(
    san_moves: List[str],
    tags: Optional[Dict[str, str]] = None,
    result: str = "*",
) -> str:
    """
    将走法列表格式化为 PGN。
    """
    t = dict(DEFAULT_TAGS)
    if tags:
        t.update({k: str(v) for k, v in tags.items()})
    t["Result"] = result

    header_lines = [f'[{k} "{v}"]' for k, v in t.items()]
    header = "\n".join(header_lines)

    # 生成带回合号的 movetext
    parts: List[str] = []
    for i, mv in enumerate(san_moves):
        move_no = i // 2 + 1
        if i % 2 == 0:
            parts.append(f"{move_no}. {mv}")
        else:
            parts.append(mv)

    movetext = " ".join(parts).strip()
    if movetext:
        movetext = f"{movetext} {result}".strip()
    else:
        movetext = result

    return f"{header}\n\n{movetext}\n"

