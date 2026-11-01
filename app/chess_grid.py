"""Recover ruled scoresheet cells. Never infer move order from text height."""
import cv2
import numpy as np

def _peaks(mask, axis, length):
    indices = np.flatnonzero(np.count_nonzero(mask, axis=axis) > length * .15)
    groups = np.split(indices, np.where(np.diff(indices) > 12)[0] + 1)
    return [float(np.median(g)) for g in groups if len(g)]

def regular_lines(ys, height):
    # Recover only small gaps supported by the repeated row spacing.
    ys=np.asarray(ys,dtype=float)
    if len(ys)<8:raise ValueError("横线证据不足。")
    gaps=np.diff(ys);step=float(np.median(gaps))
    if step<=0:raise ValueError("横线间距异常。")
    fixed=[float(ys[0])]
    for end in ys[1:]:
        gap=end-fixed[-1]
        if gap<.45*step:continue
        count=int(round(gap/step))
        if count<1 or count>3 or abs(gap/count-step)>.25*step:
            raise ValueError("横线间距不一致，请提供更平整清晰的照片。")
        start=fixed[-1]
        fixed.extend(start+gap*j/count for j in range(1,count+1))
    return fixed

def grid_order(image_rgb, items):
    image = np.asarray(image_rgb)
    scale = min(1., 1600 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,31,12)
    kernel = max(40, int(small.shape[1] * .08))
    horiz = cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,kernel),np.uint8))
    vert = cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((kernel,1),np.uint8))
    mask = cv2.dilate(horiz | vert,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("未找到表格线，请使用普通分栏模式。")
    hull = cv2.convexHull(max(contours,key=cv2.contourArea))
    quad = cv2.approxPolyDP(hull,.02*cv2.arcLength(hull,True),True).reshape(-1,2)
    if len(quad)!=4:
        for epsilon in (.03,.04,.05):
            candidate=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True).reshape(-1,2)
            if len(candidate)==4 and cv2.contourArea(candidate)>=.90*cv2.contourArea(hull):
                quad=candidate;break
    if len(quad)!=4: raise ValueError("表格边界不完整，需要辅助框选。")
    quad=quad.astype(np.float32)
    sums=quad.sum(1); dif=np.diff(quad,axis=1).ravel()
    quad=quad[[sums.argmin(),dif.argmin(),sums.argmax(),dif.argmax()]]
    w,h=1400,1450
    matrix=cv2.getPerspectiveTransform(quad,np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]))
    warped=cv2.warpPerspective(small,matrix,(w,h))
    gray=cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    xs=_peaks(cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((100,1),np.uint8)),0,h)
    ys=_peaks(cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,100),np.uint8)),1,w)
    if xs and xs[0]>15: xs.insert(0,0.)
    if xs and xs[-1]<w-20: xs.append(float(w-1))
    if ys and ys[0]>15: ys.insert(0,0.)
    if ys and ys[-1]<h-20: ys.append(float(h-1))
    if len(xs)<4 or (len(xs)-1)%3: raise ValueError("无法确认回合/白方/黑方三列结构。")
    if len(ys)<5: raise ValueError("表格行数不足。")
    ys=regular_lines(ys,h)
    steps=np.diff(ys)
    if np.max(steps)>1.6*np.median(steps) or np.min(steps)<.5*np.median(steps):
        raise ValueError("表格横线不完整，未自动生成走法，避免错位。")
    rows=len(ys)-2
    slots={}
    for item in items:
        x1,y1,x2,y2=item["bbox"]
        center=np.float32([[[((x1+x2)/2)*scale,((y1+y2)/2)*scale]]])
        x,y=cv2.perspectiveTransform(center,matrix)[0,0]
        col=int(np.searchsorted(xs,x)-1); row=int(np.searchsorted(ys,y)-1)
        if col<0 or col>=len(xs)-1 or row<1 or row>rows or col%3==0: continue
        index=((col//3)*rows+row-1)*2+(col%3-1)
        entry=dict(item,round=index//2+1,side="白方" if index%2==0 else "黑方",slot=index)
        if index in slots:
            entry['text']='?';entry['score']=0.
            entry['alternatives']=[slots[index]['text'],item['text']]
        slots[index]=entry
    if not slots: raise ValueError("表格中没有识别到走法。")
    ordered=[]
    inverse=np.linalg.inv(matrix)
    for index in range(max(slots)+1):
        panel=(index//2)//rows;row=(index//2)%rows+1;col=panel*3+1+index%2
        corners=np.float32([[[xs[col],ys[row]],[xs[col+1],ys[row+1]]]])
        original=cv2.perspectiveTransform(corners,inverse)[0]/scale
        bbox=[float(original[0,0]),float(original[0,1]),float(original[1,0]),float(original[1,1])]
        entry=slots.get(index,dict(text="?",bbox=bbox,score=0.,round=index//2+1,side="白方" if index%2==0 else "黑方",slot=index))
        entry['cell_bbox']=bbox
        ordered.append(entry)
    return ordered,dict(rows=rows,panels=(len(xs)-1)//3)
