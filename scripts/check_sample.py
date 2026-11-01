"""Run: python scripts/check_sample.py [path/to/original.jpg]"""
import io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"app"))
import chess.pgn
from chess_sequence import propose_sequence
from pgn_builder import moves_to_pgn
if len(sys.argv)>1:
    from ocr_predictor import default_predictor_with_notice,load_rgb_image
    from chess_grid import grid_order
    predictor, notice=default_predictor_with_notice()
    if notice: print(notice)
    raw=predictor.detect_and_recognize(sys.argv[1])
    cells,metadata=grid_order(load_rgb_image(sys.argv[1]),raw)
    print(metadata)
else:
    cells=json.loads((ROOT/"examples/sample-ocr-cells.json").read_text())
proposal=propose_sequence([x["text"] for x in cells])
assert proposal["moves"],proposal
game=chess.pgn.read_game(io.StringIO(moves_to_pgn(proposal["moves"])))
expected=chess.pgn.read_game(io.StringIO((ROOT/"examples/sample-reviewed.pgn").read_text()))
assert not game.errors
assert list(game.mainline_moves())==list(expected.mainline_moves())
print("Sample passed: 56 legal plies; proposed changes:",proposal["changes"])
