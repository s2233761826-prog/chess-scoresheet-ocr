import sys,os,json,time,random,math
from pathlib import Path
import numpy as np,cv2,paddle,yaml
sys.path.insert(0,'training/PaddleOCR')
from ppocr.modeling.architectures import build_model
paddle.set_device('cpu');paddle.seed(42);np.random.seed(42);random.seed(42)
cfg=yaml.safe_load(open('scripts/rec_mv3_none_bilstm_ctc_chess.yml'))['Architecture'];cfg['Head']['out_channels']=43
m=build_model(cfg);w=dict(np.load('training/run/recovered.npz'));state=m.state_dict()
missing=[]
for k,v in state.items():
 if k in w and tuple(v.shape)==w[k].shape:v.set_value(w[k])
 else:missing.append(k)
print('missing',missing,flush=True)
chars=['blank']+Path('data/dict_chess.txt').read_text().splitlines()+[' '];lookup={c:i for i,c in enumerate(chars)}
records=json.load(open('training/run/records.json'))
def image(r):
 im=cv2.imread(r['path']);h,ww=im.shape[:2];width=min(320,max(1,math.ceil(48*ww/h)));im=cv2.resize(im,(width,48)).astype('float32')/127.5-1
 out=np.zeros((3,48,320),dtype='float32');out[:,:,:width]=im.transpose(2,0,1);return out
for r in records:r['im']=image(r)
sets={s:[r for r in records if r['split']==s] for s in ('train','val','test')}
def decode(pred):
 out=[]
 for seq in pred.argmax(-1):
  prev=-1;s=''
  for t in seq:
   if t and t!=prev:s+=chars[t]
   prev=t
  out.append(s)
 return out
def evaluate(rs):
 m.eval();correct=0;errs=[]
 with paddle.no_grad():
  for i in range(0,len(rs),32):
   batch=rs[i:i+32];pred=m(paddle.to_tensor(np.stack([r['im'] for r in batch]))).numpy()
   for r,t in zip(batch,decode(pred)):
    correct+=t==r['text']
    if t!=r['text']:errs.append({'sheet':r['sheet'],'truth':r['text'],'pred':t,'path':r['path']})
 return {'accuracy':correct/len(rs),'correct':correct,'total':len(rs),'errors':errs}
report={'baseline':{s:evaluate(sets[s]) for s in ('val','test')},'epochs':[]}
Path('training/run/metrics.json').write_text(json.dumps(report));print('baseline',[(s,report['baseline'][s]['accuracy']) for s in ('val','test')],flush=True)
opt=paddle.optimizer.Adam(learning_rate=0.00001,parameters=m.parameters(),grad_clip=paddle.nn.ClipGradByGlobalNorm(5))
best=report['baseline']['val']['accuracy'];paddle.save(m.state_dict(),'training/run/best.pdparams')
for epoch in range(1,2):
 random.shuffle(sets['train']);m.train();t=time.time();losses=[]
 for i in range(0,len(sets['train']),32):
  batch=sets['train'][i:i+32];x=np.stack([r['im'] for r in batch]);x=np.clip(x+np.random.normal(0,.015,x.shape),-1,1).astype('float32')
  labels=np.zeros((len(batch),12),dtype='int32');lens=[]
  for j,r in enumerate(batch):
   ids=[lookup[c] for c in r['text']];labels[j,:len(ids)]=ids;lens.append(len(ids))
  y=m(paddle.to_tensor(x));logits=y.transpose([1,0,2]);loss=paddle.nn.functional.ctc_loss(logits,paddle.to_tensor(labels),paddle.full([len(batch)],y.shape[1],dtype='int64'),paddle.to_tensor(lens,dtype='int64'),blank=0,reduction='mean')
  loss.backward();opt.step();opt.clear_grad();losses.append(float(loss))
  if i%1024==0:print('epoch',epoch,'samples',i,'loss',float(loss),'seconds',round(time.time()-t),flush=True)
 val=evaluate(sets['val']);report['epochs'].append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':val['accuracy'],'seconds':time.time()-t})
 if val['accuracy']>best:best=val['accuracy'];paddle.save(m.state_dict(),'training/run/best.pdparams');report['best_epoch']=epoch
 Path('training/run/metrics.json').write_text(json.dumps(report));print('epoch_result',report['epochs'][-1],flush=True)
m.set_state_dict(paddle.load('training/run/best.pdparams'));report['selected']={s:evaluate(sets[s]) for s in ('val','test')};Path('training/run/metrics.json').write_text(json.dumps(report))
m.eval();static=paddle.jit.to_static(m,input_spec=[paddle.static.InputSpec(shape=[None,3,48,None],dtype='float32')]);paddle.jit.save(static,'training/run/export/inference');print('DONE',[(s,report['selected'][s]['accuracy']) for s in ('val','test')],flush=True)
