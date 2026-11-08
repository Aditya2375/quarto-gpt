"""Fixed-budget continuation study: same architecture, longer training, cosine decay.
Resume files are local-only. Browser uses JSON weights, never pickle.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from train import GPT

def run():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--chunk',type=int,default=1500);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(410)
 raw=Path('input.txt').read_text();chars=sorted(set(raw));ids={c:i for i,c in enumerate(chars)};data=torch.tensor([ids[c] for c in raw]);cut=int(.9*len(data));train,val=data[:cut],data[cut:];net=GPT(len(chars));opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.01);g=torch.Generator().manual_seed(851);vg=torch.Generator().manual_seed(920);history=[];step0=0
 def batch(part,rng):
  ix=torch.randint(len(part)-net.context-1,(24,),generator=rng);return torch.stack([part[i:i+net.context] for i in ix]),torch.stack([part[i+1:i+net.context+1] for i in ix])
 def evaluate():
  net.eval();out={}
  with torch.no_grad():
   for name,part in [('train',train),('validation',val)]:
    losses=[]
    for _ in range(20):
     x,y=batch(part,vg);losses.append(F.cross_entropy(net(x).reshape(-1,len(chars)),y.reshape(-1)).item())
    out[name]=sum(losses)/len(losses)
  net.train();return out
 state=Path('resume-refined.pt')
 if state.exists():
  s=torch.load(state,weights_only=True);assert s['total']==a.steps,'Training budget cannot change on resume';net.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);g.set_state(s['generator']);vg.set_state(s['evaluation_generator']);history=s['history'];step0=s['step']
 else:history.append(dict(step=0,learning_rate=.0007,**evaluate()))
 start=time.time();step=step0
 for step in range(step0+1,min(a.steps,step0+a.chunk)+1):
  lr=.00007+.5*(.0007-.00007)*(1+math.cos(math.pi*(step-1)/a.steps))
  for group in opt.param_groups:group['lr']=lr
  x,y=batch(train,g);loss=F.cross_entropy(net(x).reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  if step%250==0 or step==a.steps:
   ev=evaluate();history.append(dict(step=step,learning_rate=lr,**ev));print(step,ev,'seconds',round(time.time()-start,1),flush=True);torch.save(dict(total=a.steps,step=step,model=net.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),evaluation_generator=vg.get_state(),history=history),state)
 torch.save(dict(total=a.steps,step=step,model=net.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),evaluation_generator=vg.get_state(),history=history),state)
 if step==a.steps:
  net.eval();weights={k:v.detach().tolist() for k,v in net.state_dict().items() if not k.endswith('.mask')};fixtures=[]
  with torch.no_grad():
   for prompt in ['ROMEO:\n','To be, or not to be','The night is']:
    tokens=torch.tensor([[ids[c] for c in prompt]]);fixtures.append(dict(prompt=prompt,tokens=tokens[0].tolist(),logits=net(tokens)[0,-1].tolist()))
  out=dict(version=2,config=dict(vocab=len(chars),width=96,heads=4,layers=3,context=96),chars=chars,weights=weights,training=dict(seed=410,steps=a.steps,batch=24,learning_rate_max=.0007,learning_rate_min=.00007,schedule='cosine',corpus_chars=len(raw),train_chars=cut,validation_chars=len(raw)-cut,sha256=hashlib.sha256(raw.encode()).hexdigest(),history=history,parameters=sum(p.numel() for p in net.parameters()),selection='Fixed final step, no early stopping; validation monitored, not a new untouched test set.'),fixtures=fixtures)
  Path('1-refined-model.json').write_text(json.dumps(out,separators=(',',':')));print('EXPORTED',flush=True)
 else:print('SAVED resume at',step,flush=True)
if __name__=='__main__':run()
