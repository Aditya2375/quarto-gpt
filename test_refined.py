import hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train import GPT
from export_trace import reference

class RefinedTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.m=json.loads(Path('1-refined-model.json').read_text());self.net=GPT(len(self.m['chars']));s=self.net.state_dict()
  for k,v in self.m['weights'].items():s[k]=torch.tensor(v)
  self.net.load_state_dict(s);self.net.eval()
 def test_integrity(self):
  self.assertEqual(sum(p.numel() for p in self.net.parameters()),357473);self.assertEqual(self.m['training']['steps'],6000);self.assertEqual(self.m['training']['selection'],'Fixed final step, no early stopping; validation monitored, not a new untouched test set.');self.assertEqual(hashlib.sha256(Path('input.txt').read_bytes()).hexdigest(),self.m['training']['sha256']);self.assertEqual([h['step'] for h in self.m['training']['history']],list(range(0,6001,250)))
 def test_fixtures_and_silencing(self):
  with torch.no_grad():
   for f in self.m['fixtures']:np.testing.assert_allclose(self.net(torch.tensor([f['tokens']]))[0,-1].numpy(),f['logits'],atol=1e-5)
   tokens=torch.tensor([self.m['fixtures'][0]['tokens']]);normal=self.net(tokens);traced,m=reference(self.net,tokens);torch.testing.assert_close(normal,traced);changed,_=reference(self.net,tokens,(2,0));self.assertGreater(float((changed-normal).abs().max()),.01)
 def test_full_comparison(self):
  c=json.loads(Path('comparison.json').read_text());self.assertEqual(c['blocks'],1161);self.assertEqual(c['predictions'],111456);self.assertEqual([r['offset'] for r in c['results']],list(range(1003854,1003854+1161*96,96)));a=np.array([r['old'] for r in c['results']]);b=np.array([r['new'] for r in c['results']]);self.assertAlmostEqual(a.mean(),c['old_mean']);self.assertAlmostEqual(b.mean(),c['new_mean']);self.assertEqual(int((a>b).sum()),c['better']);self.assertEqual(int((b>a).sum()),c['worse']);self.assertLess(b.mean(),a.mean());g=np.random.default_rng(172);boot=np.array([g.choice(a-b,len(a),replace=True).mean() for _ in range(10000)]);np.testing.assert_allclose(np.quantile(boot,[.025,.975]),c['bootstrap95'])
  raw=Path('input.txt').read_text();ids={c:i for i,c in enumerate(self.m['chars'])};x=torch.tensor([[ids[c] for c in raw[r['offset']:r['offset']+96]] for r in c['results']]);y=torch.tensor([[ids[c] for c in raw[r['offset']+1:r['offset']+97]] for r in c['results']]);losses=[]
  with torch.no_grad():
   for i in range(0,len(x),32):losses.extend(F.cross_entropy(self.net(x[i:i+32]).transpose(1,2),y[i:i+32],reduction='none').mean(1).tolist())
  np.testing.assert_allclose(losses,b,atol=1e-6)
if __name__=='__main__':unittest.main()
