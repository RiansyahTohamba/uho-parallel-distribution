Device capacity

┌───────────┬──────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
│           │                      Value                       │                                     Implication                                     │
├───────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ GPU       │ RTX 3070 Ti Laptop, 8 GB VRAM, compute 8.6       │ bf16 + TF32 supported (no FP8). 8 GB is your binding constraint.                    │
│           │ (Ampere), 46 SMs                                 │                                                                                     │
├───────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ CPU (in   │ i9-12900H but only 6 logical / 3 cores exposed   │ Fine for tokenizing; keep dataloader_num_workers=2–4.                               │
│ WSL)      │                                                  │                                                                                     │
├───────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ RAM (in   │ 19 GiB total, 5.9 GiB available right now (13    │ The tighter-than-it-looks limit. Raise it in %UserProfile%\.wslconfig (memory=24GB) │
│ WSL)      │ GiB in use) + 8 GiB swap                         │  or close things before training.                                                   │
├───────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Disk      │ 897 GB free                                      │ Non-issue.                                                                          │
├───────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Stack     │ torch 2.14+cu130, TF 2.21 — both see the GPU     │ No transformers, datasets, or scikit-learn installed yet.                           │
└───────────┴──────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

Verdict: encoder fine-tuning is comfortable; 8B-class LLM fine-tuning is only possible in 4-bit QLoRA and is the wrong tool here anyway.

Recommended model

DeBERTa-v3-base (microsoft/deberta-v3-base, 183M params) as the primary model.

Why it fits this task specifically: ambiguity classification is short-sentence classification (requirement statements, ~15–60 tokens) with a small labeled set — usually a few hundred to a few thousand examples. In that regime a strong encoder beats a generative LLM, and DeBERTa-v3's disentangled attention + ELECTRA-style pretraining gives the best accuracy-per-parameter on exactly this shape of problem. At max_length=128, batch 32, bf16, it trains in ~2–4 GB VRAM and a full run takes minutes — so you can afford 5-fold CV and multiple seeds, which matters far more for a defensible result than model size does.

Supporting choices for the paper/comparison:

- Baseline: roberta-base — the most-cited comparison point; reviewers expect it.
- Domain-adapted: NoRBERT / BERT4RE (BERT retrained on requirements corpora). Worth including as a related-work baseline if you're targeting an RE venue.
- Ceiling run: deberta-v3-large (435M) fits in 8 GB with bf16 + batch 8 + grad accumulation 4 + 8-bit Adam. Expect +1–3 F1, ~4× the time.
- Classical floor: TF-IDF + linear SVM. Cheap, and on tiny ambiguity datasets it is sometimes within noise of BERT — which is itself a publishable finding.

Two things change the answer:

1. If your requirements are Indonesian → indobenchmark/indobert-base-p2, with microsoft/mdeberta-v3-base as the multilingual comparison. Don't useEnglish DeBERTa on Indonesian text.
2. If you have under ~300 labeled sentences → use SetFit over a sentence-transformer instead of standard fine-tuning. It's contrastive few-shot, trainsin under a minute, and reliably beats fine-tuned BERT at 8–64 examples per class.

Where an LLM does earn its place: not as the classifier, but for pseudo-labeling unlabeled requirements and for generating human-readable explanations of why a sentence is ambiguous. Use an API model for that rather than burning your 8 GB on a local 8B.

Install for the recommended path:

.venv/bin/pip install "transformers[torch]" datasets evaluate scikit-learn sentencepiece pandas

One design note worth settling before you train: whether your label set is binary (ambiguous / unambiguous) or multi-class by ambiguity type (lexical,syntactic, semantic, vague/anaphoric). The multi-class version has severe class imbalance in every published dataset I know of, so plan on macro-F1 plus stratified CV rather than accuracy. If you don't have a dataset yet, look at the ReqEval anaphoric-ambiguity shared task and the TAPHSIR/Ezzini et al.releases — verify their current availability yourself, as I'm going from memory on the specifics.

Want me to set up a baseline training script with stratified CV and macro-F1 reporting.