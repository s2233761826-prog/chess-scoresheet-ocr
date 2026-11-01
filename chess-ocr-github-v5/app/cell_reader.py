"""Read each established cell independently, including cells the detector missed."""
import cv2,numpy as np

def read_slot_images(predictor,image,items):
    rec=predictor.ocr.text_recognizer
    full=np.asarray(image);hsv=cv2.cvtColor(full,cv2.COLOR_RGB2HSV)
    black=((hsv[:,:,1]<90)&(hsv[:,:,2]<150)).astype('uint8')
    for item in items:
        box=item.get('cell_bbox',item['bbox']);x0,y0,x1,y1=box
        # Find the actual local rules rather than taking globally averaged y positions.
        margin=(y1-y0)*.4;lo=max(0,int(y0-margin));hi=min(black.shape[0],int(y1+margin))
        left=max(0,int(x0+5));right=min(black.shape[1],int(x1-5))
        if right>left:
            projection=black[lo:hi,left:right].mean(axis=1)
            ids=np.flatnonzero(projection>.35)
            groups=np.split(ids,np.where(np.diff(ids)>4)[0]+1)
            lines=[lo+float(np.median(g)) for g in groups if len(g)]
            if len(lines)>=2:
                top=min(lines,key=lambda v:abs(v-y0));bottom=min(lines,key=lambda v:abs(v-y1))
                if .65*(y1-y0)<bottom-top<1.4*(y1-y0):y0,y1=top,bottom
        dx=(x1-x0)*.025;dy=(y1-y0)*.025
        crop=image.crop((int(x0+dx),int(y0+dy),int(x1-dx),int(y1-dy)))
        a=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
        if not a.size:continue
        hsv_crop=cv2.cvtColor(a,cv2.COLOR_BGR2HSV)
        colored=((hsv_crop[:,:,1]>65)&(hsv_crop[:,:,2]<240)).astype('uint8')
        if colored.sum()>30 and colored.mean()>.008:
            # Coloured handwriting can be separated from neutral printed rules.
            yy,xx=np.where(colored)
            a=a[max(0,yy.min()-3):min(a.shape[0],yy.max()+4),max(0,xx.min()-3):min(a.shape[1],xx.max()+4)]
        norm=rec.resize_norm_img(a,320/48)[None].copy();rec.input_tensor.copy_from_cpu(norm);rec.predictor.run()
        probs=rec.output_tensors[0].copy_to_cpu()
        text,score=rec.postprocess_op(probs)[0]
        item['visual_probs']=probs[0].tolist()
        item['visual_chars']=list(rec.postprocess_op.character)
        item['detector_text']=item['text'];item['cell_text']=text
        if item['text']=='?' or float(score)>float(item.get('score',0))+.05:
            item['text']=text if text and score>.4 else '?';item['score']=float(score)
        item['needs_review']=item['text']!=text or float(score)<.90
        item['bbox']=box
    return items
