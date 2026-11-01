import cv2,numpy as np
from PIL import Image,ImageDraw

def preview(image,xrange,yrange,panels,rows):
    a=np.asarray(image);h,w=a.shape[:2];x0,x1=[int(w*x/100) for x in xrange];y0,y1=[int(h*y/100) for y in yrange]
    if x1<=x0 or y1<=y0:raise ValueError('边界范围不能为空')
    crop=image.crop((x0,y0,x1,y1));show=crop.copy();draw=ImageDraw.Draw(show);cw,ch=crop.size
    for r in range(rows+1):draw.line((0,r*ch/rows,cw,r*ch/rows),fill='red',width=2)
    for p in range(panels):
        # Default round-number column is 23% of each panel; boundaries remain visible.
        for t in (0,.23,.615,1):
            x=(p+t)*cw/panels;draw.line((x,0,x,ch),fill='red',width=2)
    return show,(x0,y0,x1,y1)

def read_cells(predictor,image,box,panels,rows):
    x0,y0,x1,y1=box;cw=(x1-x0)/panels;rh=(y1-y0)/rows;items=[]
    rec=predictor.ocr.text_recognizer
    for p in range(panels):
        for r in range(rows):
            for side,(a,b) in enumerate(((.23,.615),(.615,1))):
                rect=[int(x0+(p+a)*cw+3),int(y0+r*rh+3),int(x0+(p+b)*cw-3),int(y0+(r+1)*rh-3)]
                crop=cv2.cvtColor(np.asarray(image.crop(rect)),cv2.COLOR_RGB2BGR)
                norm=rec.resize_norm_img(crop,320/48)[None].copy();rec.input_tensor.copy_from_cpu(norm);rec.predictor.run()
                probs=rec.output_tensors[0].copy_to_cpu()
                text,score=rec.postprocess_op(probs)[0]
                items.append(dict(text=text if score>=.5 else '?',score=float(score),bbox=rect,visual_probs=probs[0].tolist(),visual_chars=list(rec.postprocess_op.character)))
    return items
