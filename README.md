# Quarto

A character-level GPT trained from scratch, with a browser writing desk, next-character probabilities, attention inspection and measured loss history. It is a learning model, not a chatbot or a source of quotations.

## Run

Serve this directory with `python -m http.server 8000`, then open http://localhost:8000. No build, API key or remote inference. The first visit downloads a 7.4 MB JSON checkpoint. Prompts stay in memory; there is no analytics or storage.

## Reproduce

1. Install Python 3.10+ and `python -m pip install -r requirements.txt`.
2. Download the exact [Tiny Shakespeare input](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) as `input.txt` in this directory. Its SHA-256 is recorded in 1-model.json and checked by the tests.
3. Run `python train.py --steps 1500 --threads 2`.
4. Run `python -m unittest -v` and `node test_transformer.cjs`.

The included checkpoint has 357,473 parameters: 3 pre-LN blocks, width 96, 4 heads, learned positions, GELU-tanh feed-forward layers, context 96, and a 65-character alphabet. AdamW, learning rate 0.0007, weight decay 0.01, clip norm 1, batch 24, initialization seed 410, separate train/evaluation generators. Training used PyTorch 2.14.1+cpu. Floating-point behavior and CPU/library differences may affect exact replay of training. Python fixture tolerance is 1e-5 for float32 CPU/backend drift. Browser inference uses double precision and matches PyTorch fixture logits within 3e-6.

The first 90% of 1,115,394 characters is training data; the last 10% is held out. Reported losses are means over 20 random batches per split at each checkpoint. Initial validation loss: 4.3020. Final validation loss: 1.9304640949. A contiguous character split does not prove leak-free generalization across Shakespeare's works. No test-set or conversational-quality claim is made.

## Browser controls

Enter 1-512 supported characters. Generate 64-512 new characters, choose temperature 0-1.5, a seed, and all/top-8/top-20 sampling. A 32-bit LCG makes seeded sampling reproducible for the same prompt/settings/browser numerical behavior. At temperature zero the first-index maximum wins. Top-k uses a threshold and includes ties at the boundary. Only the last 96 characters are used at each step. Stop preserves the partial result. Generation runs in a Web Worker so controls remain responsive.

The displayed probabilities are after temperature/top-k. During generation they belong to the character just sampled, not the next step after the displayed output. Attention shows the last query's weights in one head of the last layer. Attention is not an explanation of intent. Inputs outside the alphabet are rejected, not rewritten.

## Tests and security

Python tests cover causal masking, probability normalization, source hash/split, loss improvement and reference logits. The JavaScript test checks three independent PyTorch parity fixtures, attention normalization and seeded sampling. Browser checks cover generation, exact seeded replay, cancellation, unsupported-character and seed errors, desktop/mobile layout and runtime errors.

Static files only. No accounts, secrets, payments, uploads, tracking or dependencies in the browser. All text is written using textContent. CSP restricts scripts, workers and connections to this origin. The worker validates generation settings. The JSON checkpoint is data, not executable code. Never load an untrusted PyTorch pickle; this browser does not use pickle files.

## Sources

[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) is the public-domain text corpus used in character-model tutorials. [Karpathy's GPT lecture](https://github.com/karpathy/ng-video-lecture) is the educational reference. This is a small implementation of standard causal-transformer ideas, not an invention of the architecture.

## Browser test harness

`browser-check.cjs` ships the Chromium interaction/layout tests. Install Node.js and `npm install --no-save playwright`, then run `node browser-check.cjs`. It uses a local Chrome binary at `/usr/bin/google-chrome`; set `CHROME_PATH` for another location. Screenshots go to `/tmp`. The browser test server binds to localhost only.
