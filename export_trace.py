"""Independent PyTorch reference for browser inspection and head silencing."""
import json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from train import GPT

def reference(net,tokens,ablate=None):
 x=net.token(tokens)+net.position(torch.arange(tokens.shape[1]));matrices=[]
 for layer,b in enumerate(net.blocks):
  batch,t,d=x.shape;q,k,v=b.qkv(b.norm1(x)).chunk(3,-1);q,k,v=[z.reshape(batch,t,b.h,d//b.h).transpose(1,2) for z in (q,k,v)];p=(q@k.transpose(-2,-1)/math.sqrt(d//b.h)).masked_fill(b.mask[:,:,:t,:t]==0,float('-inf')).softmax(-1);matrices.append(p[0].tolist());a=p@v
  if ablate is not None and ablate[0]==layer:a[:,ablate[1]]=0
  a=a.transpose(1,2).contiguous().reshape(batch,t,d);x=x+b.proj(a);x=x+b.fc2(F.gelu(b.fc1(b.norm2(x)),approximate='tanh'))
 return net.head(net.norm(x)),matrices

def main():
 torch.set_num_threads(2);result=[]
 for filename in ['1-model.json','1-refined-model.json']:
  m=json.loads(Path(filename).read_text());net=GPT(len(m['chars']));state=net.state_dict()
  for k,v in m['weights'].items():state[k]=torch.tensor(v)
  net.load_state_dict(state);net.eval();fixtures=[]
  with torch.no_grad():
   for prompt in ['ROMEO:\n','To be, or not to be','The night is cold, and all the world is still.\n'*3]:
    tokens=[m['chars'].index(c) for c in prompt][-96:];x=torch.tensor([tokens]);logits,matrices=reference(net,x);torch.testing.assert_close(logits,net(x));losses=[None]+[-float(F.log_softmax(logits[0,i-1],-1)[token]) for i,token in enumerate(tokens) if i]
    fixtures.append(dict(tokens=tokens,logits=logits[0].tolist(),matrices=matrices,losses=losses))
   x=torch.tensor([[m['chars'].index(c) for c in 'ROMEO:\n']]);ablations=[]
   for layer in range(3):
    for head in range(4):
     logits,_=reference(net,x,(layer,head));ablations.append(dict(layer=layer,head=head,logits=logits[0,-1].tolist()))
  result.append(dict(file=filename,fixtures=fixtures,ablations=ablations))
 Path('trace-fixtures.json').write_text(json.dumps(result,separators=(',',':')))
if __name__=='__main__':main()
