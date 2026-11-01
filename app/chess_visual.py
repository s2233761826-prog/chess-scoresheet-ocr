"""Visual CTC likelihood for legal move candidates (no reference-game lookup)."""
import math
import numpy as np

def ctc_log_probability(probs, text, characters):
    indices={c:i for i,c in enumerate(characters) if i}
    if any(c not in indices for c in text): return -1e6
    labels=[0]
    for c in text: labels.extend([indices[c],0])
    logp=np.log(np.maximum(np.asarray(probs,dtype=float),1e-12))
    prev=np.full(len(labels),-np.inf);prev[0]=logp[0,0]
    if len(labels)>1:prev[1]=logp[0,labels[1]]
    for row in logp[1:]:
        current=np.full(len(labels),-np.inf)
        for j,label in enumerate(labels):
            v=prev[j]
            if j:v=np.logaddexp(v,prev[j-1])
            if j>1 and label and label!=labels[j-2]:v=np.logaddexp(v,prev[j-2])
            current[j]=v+row[label]
        prev=current
    return float(np.logaddexp(prev[-1],prev[-2])) if len(labels)>1 else float(prev[0])

def reread_rectangles(predictor, image_rgb, items):
    """Use the same rectangular crop + 48x320 padding as the labelled dataset."""
    import cv2
    image=cv2.cvtColor(np.asarray(image_rgb),cv2.COLOR_RGB2BGR)
    rec=predictor.ocr.text_recognizer
    chars=list(rec.postprocess_op.character)
    for item in items:
        x1,y1,x2,y2=map(int,item['bbox']);crop=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
        if not crop.size:continue
        h,w=crop.shape[:2]
        dy,dx=int(h*.04),int(w*.04)
        crop=crop[dy:h-dy,dx:w-dx]
        # Same padding width for every crop. Avoid batch-dependent recognition.
        norm=rec.resize_norm_img(crop,320/48)[None].copy()
        rec.input_tensor.copy_from_cpu(norm);rec.predictor.run()
        probs=rec.output_tensors[0].copy_to_cpu()
        text,score=rec.postprocess_op(probs)[0]
        item['initial_text']=item['text']
        item['alternate_text']=text or '?'
        item['alternate_score']=float(score)
        # Keep the initial reading unless the second crop has a clear confidence gain.
        # Both readings are retained for inspection; confidence is not a guarantee.
        if float(score)>float(item.get('score',0))+.02:
            item['text']=text or '?';item['score']=float(score)
        item['visual_probs']=probs[0].tolist();item['visual_chars']=chars
    return items
