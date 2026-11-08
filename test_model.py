import json,unittest
from pathlib import Path
import torch
from train import GPT
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.data=json.loads(Path('1-model.json').read_text());cls.model=GPT(len(cls.data['chars']));state=cls.model.state_dict();state.update({k:torch.tensor(v) for k,v in cls.data['weights'].items()});cls.model.load_state_dict(state);cls.model.eval()
    def test_causal_mask(self):
        a=torch.tensor([[1,2,3,4,5]]);b=torch.tensor([[1,2,3,20,30]])
        with torch.no_grad():self.assertTrue(torch.equal(self.model(a)[:,:3],self.model(b)[:,:3]))
    def test_fixtures(self):
        for f in self.data['fixtures']:
            with torch.no_grad():r=self.model(torch.tensor([f['tokens']]))[0,-1]
            self.assertLess(float((r-torch.tensor(f['logits'])).abs().max()),1e-5)
    def test_normalization(self):
        with torch.no_grad():r=self.model(torch.tensor([[1,2,3]])).softmax(-1)
        self.assertTrue(torch.allclose(r.sum(-1),torch.ones(1,3)))
    def test_split(self):
        t=self.data['training'];self.assertEqual(t['train_chars']+t['validation_chars'],t['corpus_chars']);self.assertLess(t['history'][-1]['validation'],t['history'][0]['validation'])
    def test_hash(self):
        import hashlib
        self.assertEqual(hashlib.sha256(Path('input.txt').read_bytes()).hexdigest(),self.data['training']['sha256'])
if __name__=='__main__':unittest.main()
