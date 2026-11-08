# Quarto, edition II

[Open the live laboratory](https://aditya2375.github.io/quarto-gpt/)

Two genuine character-level GPT checkpoints, trained from scratch. Compare writing, next-character distributions, full causal attention maps, passage surprise and a real head-silencing intervention. No chatbot wrapper or remote model. The original checkpoint remains intact.

## Run locally

```sh
python -m http.server 8000
# Open http://localhost:8000
```

The first load downloads two JSON checkpoints, about 15 MB. All inference runs in a worker. No accounts, uploads, storage, analytics or prompt transmission. CSP restricts scripts, workers and fetch to same-origin files. Text is rendered with textContent, not HTML. No PyTorch pickle is used by the browser.

## Same architecture, different training

Each checkpoint has 357,473 parameters, context 96, width 96, three pre-LayerNorm decoder blocks with four causal heads each, learned positional embeddings, GELU-tanh feed-forward layers, final LayerNorm and a 65-character output head. All weights are trained from scratch, seed 410.

Edition I: original 1,500-step constant learning rate 0.0007 checkpoint, unchanged in `1-model.json`.
Edition II: fixed 6,000 updates, cosine rate 0.0007 to 0.00007. Batch 24, AdamW weight decay 0.01, clip norm 1. Same architecture, data split and initialization. Training time and schedule change together, so this comparison does not isolate either effect.

Training source: first 90% of the 1,115,394-character Tiny Shakespeare corpus. Last 111,540 characters are validation and receive no gradient updates. Training and evaluation generators are separate. Edition II's budget was fixed in advance, no early stopping. Its final sampled validation loss is 1.6928404, versus original 1.9304641, each from 20 random batches. Do not confuse those sampled values with the deterministic evaluation below.

## Full validation comparison

`compare.py` evaluates all 1,161 non-overlapping 96-character input blocks of the existing validation split, targets shifted one character. 111,456 predictions, 83 trailing source characters unused. Corpus offsets and losses are recorded in `comparison.json`.

| Checkpoint | Mean nats / character |
|---|---:|
| Edition I | 1.93411145 |
| Edition II | 1.68587022 |

Paired mean reduction 0.24824123 nats, 95% block-bootstrap interval [0.23912922, 0.25723340], 10,000 resamples, seed 172. Edition II improves 1,090 blocks, worsens 71, ties 0.

This is the already-monitored validation split, not a new untouched test set. A contiguous excerpt split can share phrases, works and characters. The interval describes these fixed checkpoints and blocks from this corpus, not independent works or training-seed variability. Lower loss is a predictability result, not a claim of intelligence, factual accuracy or literary merit.

## Inspect actual computations

- Both manuscripts use the same opening, settings and seed. Each samples its own distribution. Seeded replay holds for fixed model/browser numerical behavior.
- Temperature zero takes the first maximum. Top-k uses a threshold and includes boundary ties. Unsupported alphabet characters are rejected, not silently rewritten.
- Inspector always analyzes the opening, not the generated continuation. It uses the last 96 characters. Changing the opening clears stale inspection values.
- All 12 causal attention heads in all layers are shown as full query/key matrices. Future columns are exactly zero. Fixed 0–1 color scale. Click/tap or use arrow keys to read an exact weight.
- Passage surprise is raw negative log probability for each character given its prefix in the inspected window, before sampling settings. First character is unscored. Nats and bits per character are displayed.
- Head silencing is a real edition-II forward-pass intervention: the selected head's weighted-value output is zeroed at every position before output projection. Distribution shift is raw KL(original || silenced). Top displayed alternatives follow the current temperature/top-k. This is not a semantic explanation of a head. Writing always uses unmodified checkpoints.
- Incremental K/V reuse is exact while learned positions stay fixed. At context 96, shifting the window requires a full re-prefill; cache is never used across a changed positional layout.

## Reproduce

Python 3.10+ with NumPy and PyTorch. Download the exact [input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

```sh
python -m pip install -r requirements.txt
python train_refined.py --steps 6000 --chunk 6000 --threads 2
python compare.py
python export_trace.py
python -m unittest -v
node test_deep.cjs
```

`--chunk 750` permits bounded runs: rerun the same command until it exports the final checkpoint. Local resume saves optimizer, training generator, evaluation generator, weights, step and history every 250 updates. Total budget cannot change mid-resume. To retrain from scratch, remove only the locally generated `resume-refined.pt`. Never load a resume file from an untrusted source. Resuming after completion safely re-exports. CPU/library differences can affect retraining; the shipped checkpoint is frozen. Original `train.py` reproduces edition I.

## Verification

Python tests check hash/split, both fixtures, model size, causal masking, normalization, silencing and recompute all 1,161 refined validation-block scores and bootstrap statistics. JavaScript compares all-token logits, every attention matrix, prefix causality, surprise and all 24 head ablations against independent PyTorch references. Max reference error is 1.14e-5 on float32 long-context logits. Tolerance 5e-5 accounts for CPU float32 versus browser double arithmetic. Cache parity is exact in 208 incremental/sliding steps across both checkpoints.

Browser tests exercise both-model generation/replay, stop and preserved partial output, unsupported input, seed validation, matrix controls/click/keyboard, head silencing, 96-character scoring, benchmark filters and desktop/320/390px layout. To run: install Playwright with Node, then `node browser-check.cjs`; uses local Chrome, `CHROME_PATH` can override its path.

## Sources

- [Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
- [Karpathy's GPT lecture](https://github.com/karpathy/ng-video-lecture)

These are compact educational implementations of standard causal-transformer ideas, not a claim to have invented the architecture. Generated output can repeat, invent words and lose coherence. It is not a quotation, factual answer or conversational assistant.
