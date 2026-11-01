"""Bounded legal sequence search. Proposals require explicit human approval."""
import chess
from chess_visual import ctc_log_probability
from Levenshtein import distance

def _cost(raw, san):
    if raw == san: return 0.
    a=raw.rstrip("+#"); b=san.rstrip("+#")
    if a == b: return .1
    return float(distance(a,b)) + .2

def propose_sequence(texts, beam_width=24, visual_items=None):
    visual_items=visual_items or []
    visual_cache={}
    def visual_cost(slot, raw, san):
        if slot>=len(visual_items):return 0.
        item=visual_items[slot]
        # Manual changes take precedence; do not reuse obsolete image evidence.
        if item.get('text')!=raw or not item.get('visual_probs'):return 0.
        key=(slot,san)
        if key not in visual_cache:
            score=ctc_log_probability(item['visual_probs'],san,item['visual_chars'])
            base_key=(slot,'__raw__')
            if base_key not in visual_cache:
                visual_cache[base_key]=ctc_log_probability(item['visual_probs'],raw,item['visual_chars'])
            # Bound the evidence so a noisy crop cannot overwhelm the chess rules.
            visual_cache[key]=max(-.5,min(2.,(visual_cache[base_key]-score)/max(1,len(san))*.12))
        return visual_cache[key]

    # (cost, board, moves, swapped rounds)
    beam=[(0.,chess.Board(),[],[])]
    for offset in range(0,len(texts),2):
        pair=texts[offset:offset+2]
        options=[(pair,False)]
        if len(pair)==2 and pair[0]!=pair[1]: options.append((pair[::-1],True))
        expanded=[]
        for cost,board,moves,swaps in beam:
            for ordered,swapped in options:
                partial=[(cost+(.8 if swapped else 0),board,moves,swaps+([offset//2+1] if swapped else []))]
                for step, raw in enumerate(ordered):
                    source_slot=offset+(len(pair)-1-step if swapped else step)
                    nxt=[]
                    for c,b,ms,ss in partial:
                        candidates=[]
                        for move in b.legal_moves:
                            san=b.san(move); delta=_cost(raw,san)
                            if raw != "?" and delta<=2.2:
                                candidates.append((delta+visual_cost(source_slot,raw,san),san,move))
                        candidates.sort(key=lambda t:(t[0],t[1]))
                        for delta,san,move in candidates[:5]:
                            nb=b.copy(stack=False); nb.push(move)
                            nxt.append((c+delta,nb,ms+[san],ss))
                    partial=sorted(nxt,key=lambda t:t[0])[:beam_width]
                    if not partial:break
                expanded.extend(partial)
        if not expanded:
            return {"moves":[],"stopped_round":offset//2+1,"alternatives":[]}
        # Preserve alternatives ending in different positions.
        kept=[];seen=set()
        for item in sorted(expanded,key=lambda t:t[0]):
            key=item[1].fen()
            if key in seen: continue
            seen.add(key);kept.append(item)
            if len(kept)>=beam_width:break
        beam=kept
    best=beam[0]
    return {"moves":best[2],"swapped_rounds":best[3],"cost":best[0],
            "alternatives":[{"cost":x[0],"moves":x[2]} for x in beam[1:4]],
            "changes":[{"round":i//2+1,"side":"白方" if i%2==0 else "黑方","OCR":a,"建议":b}
                       for i,(a,b) in enumerate(zip(texts,best[2])) if a!=b]}
