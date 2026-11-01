# Run from the project root. Training only; not needed to start the app.
import zipfile,json,base64,io,hashlib,random,re,collections
from pathlib import Path
from PIL import Image
root=Path('training/run/data');root.mkdir(parents=True,exist_ok=True)
records=[]; sheets=[];labels=collections.Counter();bad=[]
with zipfile.ZipFile('training/lable.zip') as z:
 for n in sorted(z.namelist()):
  if not n.endswith('.json'):continue
  d=json.loads(z.read(n));raw=base64.b64decode(d['imageData']);im=Image.open(io.BytesIO(raw)).convert('RGB');stem=Path(n).stem
  sheet={'id':stem,'hash':hashlib.sha256(raw).hexdigest(),'size':im.size};sheets.append(sheet)
  for i,s in enumerate(d['shapes']):
   t=s['label'].strip().replace('0-0-0','O-O-O').replace('0-0','O-O')
   if not re.fullmatch(r'(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?',t):bad.append([stem,i,t]);continue
   pts=s['points'];xs=[p[0] for p in pts];ys=[p[1] for p in pts];box=(max(0,int(min(xs))),max(0,int(min(ys))),min(im.width,int(max(xs))+1),min(im.height,int(max(ys))+1))
   if box[2]<=box[0] or box[3]<=box[1]:continue
   crop=im.crop(box);crop.thumbnail((320,64));p=root/f'{stem}_{i}.png';crop.save(p)
   records.append({'path':str(p),'text':t,'sheet':stem,'hash':sheet['hash']});labels[t]+=1
 hashes=sorted(set(s['hash'] for s in sheets));random.Random(20261101).shuffle(hashes);splits={h:('test' if i<20 else 'val' if i<40 else 'train') for i,h in enumerate(hashes)}
 for r in records:r['split']=splits[r['hash']]
Path('training/run/records.json').write_text(json.dumps(records))
print(json.dumps({'sheets':len(sheets),'unique_images':len(hashes),'crops':len(records),'split':dict(collections.Counter(r['split'] for r in records)),'unrecognized_labels':bad[:30],'bad_count':len(bad),'common':labels.most_common(15)},ensure_ascii=False))
