"""
棋规校验与校正模块（python-chess）

核心思路：
- 用 python-chess.Board() 维护局面
- 对每个识别出的走法字符串 move_text，尝试 board.parse_san(move_text)
- 若失败：
  - 枚举 board.legal_moves
  - 将每个合法走法转换为 SAN 字符串（board.san(move)）
  - 对候选 SAN 与 move_text 计算字符串距离（Levenshtein）
  - 推荐最接近的 1~3 个

注意：
- OCR 输出常会出现混淆：0 与 O，x 与 ×，空格/点号等
- 我们在候选匹配前会对字符串做“温和归一化”以提升召回
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import Levenshtein


def normalize_move_text(s: str) -> str:
    """
    用于“相似度比较”的归一化（不一定是合法 SAN）。

    目标：把 OCR 常见噪声消除掉，让候选排序更稳。
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    # 去空格
    s = s.replace(" ", "")

    # 0 与 O：王车易位最常见误差
    s = s.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    s = s.replace("0", "O") if ("O-O" in s or "O" in s) else s

    # 一些符号替换（手写/拍照常见）
    s = s.replace("×", "x")
    s = s.replace("–", "-").replace("—", "-")

    return s


@dataclass
class CorrectionIssue:
    """
    表示某一步走法解析失败，需要用户确认/修正。
    """

    index: int
    original: str
    candidates: List[str]
    corrected: Optional[str] = None


class ChessCorrector:
    def __init__(self, topk: int = 3):
        self.topk = int(topk)

    def get_candidates(self, board: chess.Board, move_text: str) -> List[str]:
        """
        基于当前局面生成候选 SAN 并按相似度排序。
        """
        target = normalize_move_text(move_text)
        cands: List[Tuple[int, str]] = []

        for mv in board.legal_moves:
            san = board.san(mv)
            dist = Levenshtein.distance(normalize_move_text(san), target)
            cands.append((dist, san))

        cands.sort(key=lambda x: (x[0], len(x[1])))
        return [s for _, s in cands[: self.topk]]

    def validate_moves(self, move_texts: List[str]) -> Tuple[List[str], List[CorrectionIssue]]:
        """
        只做校验，不与 UI 交互：
- 返回目前为止可解析通过的 corrected_moves（遇到第一处错误即停止）
- 返回 issues（第一个 issue）
        """
        board = chess.Board()
        corrected: List[str] = []
        issues: List[CorrectionIssue] = []

        for i, m in enumerate(move_texts):
            m0 = (m or "").strip()
            if not m0:
                issues.append(CorrectionIssue(index=i, original=m, candidates=self.get_candidates(board, m)))
                break

            try:
                mv = board.parse_san(m0)
                san = board.san(mv)
                board.push(mv)
                corrected.append(san)
            except Exception:
                issues.append(CorrectionIssue(index=i, original=m, candidates=self.get_candidates(board, m)))
                break

        return corrected, issues

    def apply_corrected_move(self, board: chess.Board, move_san: str) -> str:
        """
        将用户确认的 SAN 应用到棋盘，并返回标准化后的 SAN（python-chess 生成）。
        """
        mv = board.parse_san(move_san)
        san = board.san(mv)
        board.push(mv)
        return san

    def validate_and_correct_interactive(
        self,
        move_texts: List[str],
        chosen_moves: Dict[int, str],
    ) -> Tuple[List[str], List[CorrectionIssue]]:
        """
        适配 Streamlit 的“逐步校正”流程：

        - move_texts: OCR 原始结果
        - chosen_moves: 用户对某些 index 已经选择/输入的修正 SAN（键为 index）

        返回：
        - corrected_moves: 已确认并应用到棋盘的 SAN 列表（从头开始连续的）
        - issues: 若下一步出现错误，给出 issue（供 UI 渲染选择）；若无错误则为空

        设计要点：
        - Streamlit 是无状态刷新；我们把“用户已经修正过哪些步”放在 session_state 里，
          每次 rerun 都能继续从下一步开始校验。
        """
        board = chess.Board()
        corrected: List[str] = []
        issues: List[CorrectionIssue] = []

        for i, m in enumerate(move_texts):
            if i in chosen_moves and chosen_moves[i]:
                # 用户已为该步提供了 SAN（可能来自候选或手动输入）
                try:
                    san = self.apply_corrected_move(board, chosen_moves[i])
                    corrected.append(san)
                    continue
                except Exception:
                    # 用户输入仍不合法：给出候选并停止，让 UI 重新选择
                    issues.append(CorrectionIssue(index=i, original=m, candidates=self.get_candidates(board, m)))
                    break

            # 尚未修正，尝试直接解析 OCR 文本
            m0 = (m or "").strip()
            try:
                mv = board.parse_san(m0)
                san = board.san(mv)
                board.push(mv)
                corrected.append(san)
            except Exception:
                issues.append(CorrectionIssue(index=i, original=m, candidates=self.get_candidates(board, m)))
                break

        return corrected, issues

