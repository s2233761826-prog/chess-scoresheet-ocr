# Run from the project root. Training only; not needed to start the app.
import paddle,pickle,numpy as np
p='models/finetuned/rec_chess_infer/inference'
info=pickle.load(open(p+'.pdiparams.info','rb'))
paddle.enable_static();exe=paddle.static.Executor(paddle.CPUPlace());program,feeds,fetch=paddle.static.load_inference_model(p,exe)
scope=paddle.static.global_scope();weights={}
for name,meta in info.items():
 v=scope.find_var(name)
 if v is not None:weights[meta['structured_name']]=np.array(v.get_tensor())
np.savez('training/run/recovered.npz',**weights);print('recovered',len(weights))
