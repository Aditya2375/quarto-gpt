"""Small character GPT, trained from scratch. CPU-friendly, no hosted models."""
import argparse, hashlib, json, math, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,d,h,context):
        super().__init__();self.h=h;self.norm1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.norm2=nn.LayerNorm(d);self.fc1=nn.Linear(d,4*d);self.fc2=nn.Linear(4*d,d)
        self.register_buffer('mask',torch.tril(torch.ones(context,context)).view(1,1,context,context))
    def forward(self,x,return_attention=False):
        b,t,d=x.shape;q,k,v=self.qkv(self.norm1(x)).chunk(3,dim=-1);q,k,v=[z.reshape(b,t,self.h,d//self.h).transpose(1,2) for z in (q,k,v)];w=(q@k.transpose(-2,-1))/math.sqrt(d//self.h);w=w.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf')).softmax(-1);a=(w@v).transpose(1,2).contiguous().reshape(b,t,d);x=x+self.proj(a);x=x+self.fc2(F.gelu(self.fc1(self.norm2(x)),approximate='tanh'));return (x,w) if return_attention else x
class GPT(nn.Module):
    def __init__(self,vocab,d=96,h=4,layers=3,context=96):
        super().__init__();self.context=context;self.token=nn.Embedding(vocab,d);self.position=nn.Embedding(context,d);self.blocks=nn.ModuleList([Block(d,h,context) for _ in range(layers)]);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,vocab)
    def forward(self,idx):
        t=idx.shape[1];x=self.token(idx)+self.position(torch.arange(t,device=idx.device));
        for block in self.blocks:x=block(x)
        return self.head(self.norm(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1500);ap.add_argument('--threads',type=int,default=2);args=ap.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(410)
    raw=Path('input.txt').read_text();chars=sorted(set(raw));encode={c:i for i,c in enumerate(chars)};data=torch.tensor([encode[c] for c in raw]);cut=int(.9*len(data));train,val=data[:cut],data[cut:];model=GPT(len(chars));optim=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01);g=torch.Generator().manual_seed(851);vg=torch.Generator().manual_seed(920);history=[]
    def batch(data,g):
        ix=torch.randint(len(data)-model.context-1,(24,),generator=g);return torch.stack([data[i:i+model.context] for i in ix]),torch.stack([data[i+1:i+model.context+1] for i in ix])
    def evaluate():
        model.eval();result={}
        with torch.no_grad():
            for name,part in [('train',train),('validation',val)]:
                losses=[]
                for _ in range(20):
                    x,y=batch(part,vg);losses.append(F.cross_entropy(model(x).reshape(-1,len(chars)),y.reshape(-1)).item())
                result[name]=sum(losses)/len(losses)
        model.train();return result
    start=time.time();initial=evaluate();history.append(dict(step=0,**initial));print('initial',initial,flush=True)
    for step in range(1,args.steps+1):
        x,y=batch(train,g);loss=F.cross_entropy(model(x).reshape(-1,len(chars)),y.reshape(-1));optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
        if step%250==0 or step==args.steps:
            ev=evaluate();history.append(dict(step=step,**ev));torch.save(model.state_dict(),'checkpoint.pt');print(step,ev,round(time.time()-start,1),flush=True)
    model.eval();state={k:v.detach().tolist() for k,v in model.state_dict().items() if not k.endswith('.mask')};fixtures=[]
    with torch.no_grad():
        for prompt in ['ROMEO:\n','To be, or not to be','The night is']:
            tokens=torch.tensor([[encode[c] for c in prompt]]);logits=model(tokens)[0,-1].tolist();fixtures.append(dict(prompt=prompt,tokens=tokens[0].tolist(),logits=logits))
    payload=dict(version=1,config=dict(vocab=len(chars),width=96,heads=4,layers=3,context=96),chars=chars,weights=state,training=dict(seed=410,steps=args.steps,batch=24,learning_rate=.0007,corpus_chars=len(raw),train_chars=cut,validation_chars=len(raw)-cut,sha256=hashlib.sha256(raw.encode()).hexdigest(),history=history,parameters=sum(p.numel() for p in model.parameters())),fixtures=fixtures)
    Path('1-model.json').write_text(json.dumps(payload,separators=(',',':')));torch.save(model.state_dict(),'checkpoint.pt');print('done',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
