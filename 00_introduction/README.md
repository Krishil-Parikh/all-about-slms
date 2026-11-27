# Module 00 · Introduction to Small Language Models

Eight notebooks that take you from *"what is a language model?"* to training, saving and quantizing one you built yourself. No pretrained weights are used anywhere.

Read [`../intro-to-slms.md`](../intro-to-slms.md) first for the big picture, then work through the notebooks in order.

## Lessons

| # | Notebook | What you build | Libraries |
|:-:|:--|:--|:--|
| 00 | [What is a Small Language Model?](00_what_is_an_slm.ipynb) | A count-based bigram LM: sentence scoring with the chain rule, log-probabilities, greedy vs sampled decoding, temperature, perplexity and smoothing. Then an exact GPT parameter counter, checked against GPT-2 small, plus memory and compute budgets that define "small" | NumPy, Pandas, Matplotlib |
| 01 | [Anatomy of an SLM](01_anatomy_of_an_slm.ipynb) | A decoder-only Transformer forward pass: embeddings, sinusoidal positions, Q/K/V, scaled dot-product and causal attention, multi-head attention, FFN, LayerNorm, logits | NumPy, Pandas, Matplotlib |
| 02 | [Tokenization](02_tokenization.ipynb) | Character, word and subword tokenization; Byte Pair Encoding trained, encoded and decoded from scratch; byte-level tokenization and vocabulary trade-offs | NumPy, Pandas, Matplotlib |
| 03 | [Training data for SLMs](03_training_data_for_slms.ipynb) | A corpus pipeline: cleaning, exact and near-duplicate detection, quality filters, document-level splits, JSONL + manifest export | Pandas, Matplotlib |
| 04 | [From text to training sequences](04_from_text_to_training_sequences.ipynb) | A BPE tokenizer trained on the training split, `<eos>`/`<pad>` tokens, input/target shifting, padding masks, packing | NumPy, Pandas, Matplotlib |
| 05 | [Next-token prediction and the LM objective](05_next_token_prediction_and_language_modeling_objective.ipynb) | Logits, stable softmax, negative log-likelihood, cross-entropy, masked loss and perplexity, all by hand | NumPy, Matplotlib |
| 06 | [A tiny language model with PyTorch](06_building_a_tiny_language_model_with_pytorch.ipynb) | A small character-level Transformer with causal self-attention, trained with a real training loop, learning curves and generation | PyTorch |
| 07 | [The SLM training lifecycle, end to end](07_the_slm_training_lifecycle_end_to_end.ipynb) | Every lifecycle stage in one place: data cleaning and dedup, a tokenizer, initialization at log(V), **hand-written backprop with a gradient check**, AdamW and warmup + cosine from scratch, best-checkpoint saving, overfitting analysis, a context-length sweep, fine-tuning and forgetting, save/reload, int8 quantization | NumPy, Pandas, Matplotlib |

## How the notebooks connect

```text
00 ─ what a language model is, and what "small" means
01 ─ what a Transformer computes (forward pass, random weights)
02 ─ how text becomes tokens (BPE)
03 ─ raw documents ──► prepared_corpus/*.jsonl ─┐
04 ─ prepared_corpus ──► tokenized_dataset/*    ◄┘   (04 reads what 03 writes)
05 ─ the training objective: cross-entropy and perplexity
06 ─ a trainable Transformer in PyTorch
07 ─ the full lifecycle, with nothing hidden behind a framework
```

Notebooks **03 and 04 share files on disk**. Run 03 before 04 so that `prepared_corpus/` holds the current splits. All other notebooks are self-contained.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r 00_introduction/requirements.txt
jupyter lab
```

Every notebook except 06 needs only NumPy, Pandas and Matplotlib, and runs in seconds on a laptop CPU. Notebook 06 also needs PyTorch. It uses a GPU if one is available, and the CPU otherwise.

## Folder contents

```text
00_introduction/
├── 00_what_is_an_slm.ipynb
├── 01_anatomy_of_an_slm.ipynb
├── 02_tokenization.ipynb
├── 03_training_data_for_slms.ipynb
├── 04_from_text_to_training_sequences.ipynb
├── 05_next_token_prediction_and_language_modeling_objective.ipynb
├── 06_building_a_tiny_language_model_with_pytorch.ipynb
├── 07_the_slm_training_lifecycle_end_to_end.ipynb
├── prepared_corpus/        # written by 03: train/validation/test JSONL + dataset_manifest.json
├── tokenized_dataset/      # written by 04: token IDs per split + tokenizer.json
├── requirements.txt
└── README.md
```

## After this module

**Module 01 · Tokenization in depth**: byte-level BPE on a real corpus, pre-tokenization, and vocabulary-size experiments. See the roadmap in [`intro-to-slms.md`](../intro-to-slms.md#️-10-the-learning-roadmap).
