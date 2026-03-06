# Training Details - Persistent Embed

Complete documentation of training methodology, data, and infrastructure.

---

## Architecture

Persistent Embed uses a bi-encoder architecture based on the Sentence-BERT framework, extended with:

- **Base:** BERT-style encoder (bert-base-multilingual-cased foundation)
- **Pooling:** Mean pooling over token embeddings
- **Normalisation:** L2 normalisation of output embeddings
- **Context:** 512 tokens (50M/110M), 8192 tokens (350M, with RoPE)

### Model Sizes

| Model | Layers | Hidden | Heads | FFN | Parameters |
|---|---|---|---|---|---|
| 50M | 6 | 384 | 12 | 1536 | 50M |
| 110M | 12 | 768 | 12 | 3072 | 110M |
| 350M | 24 | 1024 | 16 | 4096 | 350M |

---

## Training Objective

We use **MultipleNegativesRankingLoss** (MNRL) with hard negative mining:

```
Loss = -log(exp(sim(q, p+) / τ) / Σ exp(sim(q, ni) / τ))
```

Where:
- `q` = query embedding
- `p+` = positive (matching) passage embedding
- `ni` = negative (non-matching) passage embeddings
- `τ` = temperature (0.05)

### Hard Negative Mining

We mine hard negatives using BM25 retrieval - for each query, the top-ranked BM25 results that are not the true positive become hard negatives. This forces the model to learn fine-grained semantic distinctions rather than simple lexical matching.

---

## Training Data

### Multilingual Text Pairs

| Dataset | Languages | Pairs | Domain |
|---|---|---|---|
| mMARCO | 13 languages incl. Hindi | 8.8M | Web search |
| MIRACL | 18 languages incl. 6 Indian | 726K | Wikipedia |
| Mr. TyDi | 11 languages | 188K | Wikipedia |
| IndicNLP | 6 Indian languages | 2M | News, Wikipedia |
| Custom curated | 6 Indian languages | 500K | Diverse |

### Indian Language Sources

We partner with IIIT Hyderabad LTRC and collect from:
- **Wikipedia** dumps in all 10 Indian languages
- **Common Crawl** filtered Indian language segments
- **News corpora** — major Indian newspapers with open archives
- **Legal documents** — Indian court judgements (Creative Commons)
- **Academic papers** — Indian research in regional languages

---

## Training Infrastructure

```
Primary:   Google TPU v3-8 (via TRC grant)
Secondary: 8x A100 80GB (via HuggingFace grant)
Framework: PyTorch + HuggingFace Transformers
Distributed: DeepSpeed ZeRO-2 / PyTorch XLA (TPU)
Tracking:  Weights & Biases
```

### Hyperparameters

| Parameter | 50M | 110M | 350M |
|---|---|---|---|
| Batch size | 1024 | 512 | 256 |
| Learning rate | 2e-5 | 2e-5 | 1e-5 |
| Warmup steps | 1000 | 2000 | 5000 |
| Training steps | 100K | 200K | 500K |
| Max sequence length | 512 | 512 | 8192 |
| Temperature | 0.05 | 0.05 | 0.05 |
| Optimizer | AdamW | AdamW | AdamW |
| LR schedule | Cosine | Cosine | Cosine |

---

## Evaluation

### MTEB Tasks Evaluated

- **Retrieval:** MSMARCO, BEIR suite (15 datasets)
- **Semantic Textual Similarity:** STS12-17, STSBenchmark
- **Clustering:** ArXiv, Reddit, StackExchange
- **Classification:** AmazonReviews, Banking77, ToxicConversations
- **Reranking:** MindSmallReranking, SciDocsRR
- **Bitext Mining:** BUCC, Tatoeba

### Indian Language Tasks (Custom)

- **INDIC-STS:** Semantic textual similarity in 6 Indian languages
- **INDIC-Retrieval:** Information retrieval in Hindi, Telugu, Tamil
- **Cross-lingual:** English query → Indian language document retrieval
- **Code-switching:** Mixed Hindi-English (Hinglish) retrieval

---

## Reproducibility

All training runs are fully reproducible:

```bash
# Reproduce 50M training run
git clone https://github.com/persistent-research/embed
cd embed
git checkout v1.0.0-50m

pip install -r requirements.txt
python data/download.py --config configs/50m_base.yaml
python data/preprocess.py --config configs/50m_base.yaml

python persistent_embed/train.py \
  --config configs/50m_base.yaml \
  --seed 42 \
  --output_dir ./reproduced_50m

# Evaluate
python evaluation/run_eval.py \
  --model ./reproduced_50m \
  --benchmark all
```

Training logs, W&B runs, and checkpoint hashes are published in `evaluation/results/`.

---

*Questions about training? Open an issue or email research@persistentresearch.in*
