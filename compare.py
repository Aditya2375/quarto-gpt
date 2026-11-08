"""Frozen comparison over all non-overlapping 96-character blocks of validation.
This is the existing monitored validation split, not an untouched test set.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train import GPT

def main():
 torch.set_num_threads(2);raw=Path('input.txt').read_text();old=json.loads(Path('1-model.json').read_text());new=json.loads(Path('1-refined-model.json').read_text());ids={c:i for i,c in enumerate(old['chars'])};cut=old['training']['train_chars'];data=torch.tensor([ids[c] for c in raw[cut:]]);starts=list(range(0,len(data)-96,96));blocks=[]
 for payload in [old,new]:
  net=GPT(len(ids));state=net.state_dict()
  for k,v in payload['weights'].items():state[k]=torch.tensor(v)
  net.load_state_dict(state);net.eval();scores=[]
  with torch.no_grad():
   for at in range(0,len(starts),32):
    ix=starts[at:at+32];x=torch.stack([data[i:i+96] for i in ix]);y=torch.stack([data[i+1:i+97] for i in ix]);loss=F.cross_entropy(net(x).transpose(1,2),y,reduction='none').mean(1);scores.extend(loss.tolist())
  blocks.append(scores)
 delta=np.array(blocks[0])-np.array(blocks[1]);g=np.random.default_rng(172);boot=np.array([g.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
 summary=dict(scope='Existing monitored validation split, not a new untouched test set. Every complete non-overlapping 96-character input block, target shifted by one. No selection on this comparison.',source_sha256=hashlib.sha256(raw.encode()).hexdigest(),split_start=cut,validation_chars=len(data),context=96,blocks=len(starts),predictions=len(starts)*96,unused_trailing_chars=len(data)-(starts[-1]+97),old_mean=float(np.mean(blocks[0])),new_mean=float(np.mean(blocks[1])),mean_gain=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_seed=172,bootstrap_resamples=10000,better=int((delta>0).sum()),same=int((delta==0).sum()),worse=int((delta<0).sum()),results=[dict(offset=cut+s,old=a,new=b) for s,a,b in zip(starts,*blocks)])
 Path('comparison.json').write_text(json.dumps(summary,separators=(',',':')));print({k:v for k,v in summary.items() if k!='results'})
if __name__=='__main__':main()
