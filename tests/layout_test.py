import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from chess_grid import grid_order
from PIL import Image,ImageDraw
for panels in (1,2,3):
 im=Image.new('RGB',(panels*260+80,740),'white');d=ImageDraw.Draw(im);items=[]
 for j in range(22):d.line((40,40+j*30,40+panels*260,40+j*30),fill='black',width=3)
 for p in range(panels):
  for x in (0,60,160,260):d.line((40+p*260+x,40,40+p*260+x,670),fill='black',width=3)
  for r in range(20):
   for side in (0,1):
    x=40+p*260+75+side*100;y=75+r*30
    if (p,r,side)!=(0,4,0):items.append(dict(text='e4',score=1,bbox=[x,y,x+55,y+18]))
 out,m=grid_order(im,items);assert len(out)==panels*40,(panels,m,len(out));assert out[8]['text']=='?'
 print('PASS',panels,'panels',len(out),'slots; missing cell retained')
