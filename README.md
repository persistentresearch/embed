# Persistent Embed

<div align="center">

<img src="assets/persistent-research-logo.png" alt="Persistent Research" width="120"/>

**Open-source multilingual embedding models from Persistent Research.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-black.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-persistent--research-yellow)](https://huggingface.co/persistent-research)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-red)](https://arxiv.org)
[![Discord](https://img.shields.io/badge/Discord-Join_Community-5865F2)](https://discord.gg/persistentresearch)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub_Sponsors-EA4AAA)](https://github.com/sponsors/persistent-research)
[![Open Collective](https://img.shields.io/badge/Donate-Open_Collective-blue)](https://opencollective.com/persistent-embed)

*Forever open source. Community powered. Indian languages first.*

[Models](#models) · [Quickstart](#quickstart) · [Training](#training) · [Evaluation](#evaluation) · [Community](#community) · [Roadmap](#roadmap)

</div>

---

## What is Persistent Embed?

Persistent Embed is a series of forever open-source multilingual text embedding models trained and maintained by Persistent Research. Built specifically with Indian languages as first-class citizens, Persistent Embed provides high-quality semantic embeddings for Hindi, Telugu, Tamil, Kannada, Bengali, Marathi, and 30+ world languages alongside English.

New model sizes are released every 6 months. All weights, training code, evaluation frameworks, and data pipelines are open. Always.

### Why Persistent Embed?

Current state-of-the-art embedding models treat Indian languages as afterthoughts. Models like `text-embedding-ada-002`, `multilingual-e5`, and `BGE-m3` are trained primarily on English and European language data — performance on Indian languages lags significantly behind English performance on every benchmark.

Persistent Embed is built differently. Indian language corpora are treated as primary, not supplementary. We partner with Indian universities and language research institutions to source and curate high-quality training data across India's linguistic diversity.

---

## Models

| Model | Parameters | Dimensions | Context | Languages | Status |
|---|---|---|---|---|---|
| `persistent-embed-50m` | 50M | 384 | 512 tokens | 35+ | 🔄 Training |
| `persistent-embed-110m` | 110M | 768 | 512 tokens | 35+ | 🔄 Planned |
| `persistent-embed-350m` | 350M | 1024 | 8192 tokens | 35+ | 🔄 Planned |

### Indian Language Coverage

| Language | Script | Training Data Size | Benchmark Status |
|---|---|---|---|
| Hindi | Devanagari | Primary | ✅ Evaluated |
| Telugu | Telugu | Primary | ✅ Evaluated |
| Tamil | Tamil | Primary | ✅ Evaluated |
| Kannada | Kannada | Primary | ✅ Evaluated |
| Bengali | Bengali | Primary | ✅ Evaluated |
| Marathi | Devanagari | Primary | ✅ Evaluated |
| Gujarati | Gujarati | Secondary | 🔄 Planned |
| Punjabi | Gurmukhi | Secondary | 🔄 Planned |
| Malayalam | Malayalam | Secondary | 🔄 Planned |
| Urdu | Nastaliq | Secondary | 🔄 Planned |

---

## Quickstart

### Install

```bash
pip install persistent-embed
```

### Basic Usage

```python
from persistent_embed import PersistentEmbed

# Load model
model = PersistentEmbed("persistent-research/persistent-embed-50m")

# Embed single sentence
embedding = model.encode("Hello, world!")
print(embedding.shape)  # (384,)

# Embed batch
sentences = [
    "नमस्ते दुनिया",           # Hindi
    "హలో ప్రపంచం",              # Telugu
    "வணக்கம் உலகம்",           # Tamil
    "Hello, world!",            # English
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 384)
```

### Semantic Search

```python
from persistent_embed import PersistentEmbed
import numpy as np

model = PersistentEmbed("persistent-research/persistent-embed-50m")

# Documents in mixed languages
documents = [
    "Artificial intelligence is transforming industries worldwide.",
    "कृत्रिम बुद्धिमत्ता उद्योगों को बदल रही है।",  # Hindi translation
    "Machine learning requires large amounts of data.",
    "Deep learning models need powerful GPUs for training.",
]

# Query in any language
query = "AI needs data and compute"

# Encode
doc_embeddings = model.encode(documents)
query_embedding = model.encode(query)

# Cosine similarity
similarities = np.dot(doc_embeddings, query_embedding) / (
    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
)

# Ranked results
ranked = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"{score:.4f}: {doc[:60]}")
```

### HuggingFace Transformers (direct)

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "persistent-research/persistent-embed-50m"
)
model = AutoModel.from_pretrained(
    "persistent-research/persistent-embed-50m"
)

def encode(texts):
    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded)
    # Mean pooling
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings

embeddings = encode(["Hello world", "नमस्ते दुनिया"])
print(embeddings.shape)  # (2, 384)
```

### ONNX (Edge / Mobile)

```python
from persistent_embed.onnx import PersistentEmbedONNX

# Quantised ONNX model - runs on CPU, mobile, edge
model = PersistentEmbedONNX("persistent-embed-50m-int8")
embedding = model.encode("Hello world")
```

---

## Repository Structure

```
persistent-embed/
│
├── configs/                    # Training configuration files
│   ├── 50m_base.yaml           # 50M parameter config
│   ├── 110m_base.yaml          # 110M parameter config
│   └── 350m_base.yaml          # 350M parameter config
│
├── data/                       # Data pipeline
│   ├── download.py             # Dataset download scripts
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── curate.py               # Quality filtering
│   ├── dedup.py                # Deduplication
│   └── README.md               # Data documentation
│
├── persistent_embed/           # Main Python package
│   ├── __init__.py
│   ├── model.py                # Model architecture
│   ├── tokenizer.py            # Tokeniser utilities
│   ├── encode.py               # Encoding interface
│   ├── train.py                # Training entry point
│   ├── onnx.py                 # ONNX export and inference
│   └── utils.py                # Utility functions
│
├── training/                   # Training infrastructure
│   ├── trainer.py              # Main trainer class
│   ├── loss.py                 # Contrastive loss functions
│   ├── scheduler.py            # Learning rate schedulers
│   ├── data_loader.py          # Training data loader
│   └── distributed.py          # Multi-GPU/TPU training
│
├── evaluation/                 # Evaluation framework
│   ├── run_eval.py             # Main evaluation runner
│   ├── benchmarks/             # Benchmark implementations
│   │   ├── mteb.py             # MTEB benchmark
│   │   ├── indic_bench.py      # Indian language benchmark
│   │   └── custom.py           # Custom evaluation tasks
│   └── results/                # Published results
│       └── README.md
│
├── scripts/                    # Utility scripts
│   ├── convert_to_onnx.py      # ONNX export
│   ├── quantize.py             # Post-training quantisation
│   ├── upload_to_hf.py         # HuggingFace upload
│   └── benchmark.py            # Performance benchmarking
│
├── tests/                      # Unit and integration tests
│   ├── test_model.py
│   ├── test_encode.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── docs/                       # Documentation
│   ├── training.md             # Training guide
│   ├── evaluation.md           # Evaluation guide
│   ├── contributing.md         # Contribution guide
│   └── architecture.md        # Architecture deep dive
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Continuous integration
│   │   ├── eval.yml            # Automated evaluation
│   │   └── release.yml         # Model release pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── MODEL_CARD.md               # HuggingFace model card
├── TRAINING_DETAILS.md         # Full training documentation
├── CITATION.md                 # How to cite this work
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
├── CODE_OF_CONDUCT.md          # Community standards
├── SECURITY.md                 # Security policy
├── LICENSE                     # Apache 2.0
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

---

## Training

Persistent Embed uses a contrastive learning approach with hard negative mining. Full training details in [TRAINING_DETAILS.md](TRAINING_DETAILS.md).

### Quick Training Run

```bash
# Clone repository
git clone https://github.com/persistent-research/embed
cd embed

# Install dependencies
pip install -r requirements.txt

# Download and prepare training data
python data/download.py --config configs/50m_base.yaml
python data/preprocess.py --config configs/50m_base.yaml

# Run training (single GPU)
python persistent_embed/train.py \
  --config configs/50m_base.yaml \
  --output_dir ./checkpoints/50m

# Run training (multi-GPU with DeepSpeed)
deepspeed persistent_embed/train.py \
  --config configs/50m_base.yaml \
  --deepspeed_config configs/ds_config.json \
  --output_dir ./checkpoints/50m
```

### TPU Training (Google TRC)

```bash
# On TPU VM
pip install torch_xla
python persistent_embed/train.py \
  --config configs/50m_base.yaml \
  --tpu \
  --tpu_cores 8 \
  --output_dir gs://your-bucket/checkpoints/50m
```

---

## Evaluation

We evaluate Persistent Embed on MTEB (Massive Text Embedding Benchmark) plus our own Indian language evaluation suite.

```bash
# Run full MTEB evaluation
python evaluation/run_eval.py \
  --model persistent-research/persistent-embed-50m \
  --benchmark mteb \
  --output_dir ./results

# Run Indian language evaluation
python evaluation/run_eval.py \
  --model persistent-research/persistent-embed-50m \
  --benchmark indic \
  --output_dir ./results

# Run custom evaluation on your data
python evaluation/run_eval.py \
  --model persistent-research/persistent-embed-50m \
  --benchmark custom \
  --data_path ./your_eval_data.json \
  --output_dir ./results
```

Results are published in `evaluation/results/` and updated with each model release.

---

## Roadmap

### v1.0 — Foundation (Month 1–6)
- [ ] persistent-embed-50m (384 dimensions)
- [ ] 6 Indian languages — primary support
- [ ] MTEB evaluation published
- [ ] Indian language benchmark published
- [ ] pip package released
- [ ] ONNX export for on-device use
- [ ] arXiv paper submitted

### v1.5 — Expansion (Month 6–12)
- [ ] persistent-embed-110m (768 dimensions)
- [ ] 10 Indian languages
- [ ] Extended context (8K tokens) on 110M model
- [ ] Improved hard negative mining
- [ ] Community fine-tuning guide published

### v2.0 — Frontier (Month 12–18)
- [ ] persistent-embed-350m (1024 dimensions)
- [ ] All 22 scheduled Indian languages
- [ ] Domain-specific variants (legal, medical, code)
- [ ] Multilingual instruction-following embeddings
- [ ] Second research paper

---

## Community

Persistent Embed is community-powered. We cannot do this without you.

### Contribute Compute
We need GPU/TPU time to train models. If you have spare compute:
- Email: research@persistentresearch.in
- Subject: "Compute contribution - Persistent Embed"

### Contribute Data
High-quality Indian language text pairs for training:
- Open an issue: "Data contribution"
- Include language, domain, and approximate size

### Contribute Code
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Pick an issue labelled `good first issue`
- Submit a pull request

### Contribute Financially
This project runs on donated compute and community funding:
- [GitHub Sponsors](https://github.com/sponsors/persistent-research)
- [Open Collective](https://opencollective.com/persistent-embed)

### Join the Conversation
- [Discord](https://discord.gg/persistentresearch) - real-time discussion
- [GitHub Discussions](https://github.com/persistent-research/embed/discussions) - async discussion
- [HuggingFace Community](https://huggingface.co/persistent-research) - model discussions

---

## Citation

```bibtex
@misc{persistentresearch2026embed,
  title        = {Persistent Embed: Open-Source Multilingual Embeddings
                  with Indian Language Focus},
  author       = {Persistent Research},
  year         = {2026},
  publisher    = {Persistent Research},
  url          = {https://github.com/persistent-research/embed},
  note         = {Model weights at huggingface.co/persistent-research}
}
```

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).

Free for personal, academic, and commercial use.

---

## Acknowledgements

Persistent Embed builds on the open source work of the global AI research community. We specifically thank:

- **EleutherAI** - for open source compute access and community support
- **HuggingFace** - for the Transformers ecosystem and compute grants
- **Google TRC** - for TPU research access
- **IIIT Hyderabad LTRC** - for Indian language data partnerships
- **The global open source NLP community** - for the foundational research this builds on

---

<div align="center">

**Persistent Research**

Built in India. Built for the world.

[persistentresearch.in](https://persistentresearch.in) · [research@persistentresearch.in](mailto:research@persistentresearch.in)

</div>
