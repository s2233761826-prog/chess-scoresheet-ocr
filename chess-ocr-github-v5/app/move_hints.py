"""Local suggestions; never modifies the game or searches a replacement game."""
import chess
from Levenshtein import distance
from chess_visual import ctc_log_probability

def normalize(text):
    t=str(text or '').strip().replace('×','x').replace('–','-')
    if t.startswith('0-0'):t=t.replace('0','O')
    return t

def prefix_board(moves,index):
    b=chess.Board()
    for i,move in enumerate(moves[:index]):
        try:b.push_san(normalize(move))
        except ValueError:return None,i
    return b,None

def suggestions(moves,index,item=None,limit=4):
    item=item or {};board,blocked=prefix_board(moves,index)
    if board is None:return [],blocked
    readings=list(dict.fromkeys(normalize(x) for x in [moves[index],item.get('cell_text'),item.get('detector_text')] if x and str(x)!='?'))
    probs=item.get('visual_probs');chars=item.get('visual_chars')
    candidates=[]
    for move in board.legal_moves:
        san=board.san(move)
        spelling=min((distance(san.rstrip('+#'),r.rstrip('+#')) for r in readings),default=0)
        visual=None
        if probs is not None and chars:
            visual=-ctc_log_probability(probs,san,chars)/max(1,len(san))
        candidates.append(dict(move=san,spelling=spelling,visual=visual))
    if not readings and probs is None:return [],None
    # Image evidence ranks candidates alongside textual similarity, never game strength.
    base=min((c['visual'] for c in candidates if c['visual'] is not None),default=0)
    for c in candidates:
        c['rank']=c['spelling']+.35*min(12,max(0,c['visual']-base)) if c['visual'] is not None else c['spelling']
    candidates.sort(key=lambda c:(c['rank'],c['spelling'],c['move']))
    return candidates[:limit],None
