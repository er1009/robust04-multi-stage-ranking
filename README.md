# Multi-Stage Document Ranking Pipeline for TREC ROBUST04

A high-performance document ranking system combining **lexical**, **learned sparse**, **dense**, **neural**, and **LLM-based** retrieval and reranking methods. Achieves **MAP = 0.41** on the TREC ROBUST04 benchmark through progressive refinement -- each stage adds capabilities the previous cannot provide.

## Highlights

- **6-way hybrid retrieval**: BM25+RM3, Dense (BGE), SPLADE, each with Query2Doc expansion
- **3-stage neural cascade**: Bi-Encoder → Cross-Encoder Ensemble → MonoT5 (1.7B+ total params)
- **LLM reranking cascade**: GPT-4o-mini (sliding window) → GPT-5 (precision refinement)
- **Reciprocal Rank Fusion** at two levels for robust score combination
- **Contextual chunking** with MaxP aggregation for long documents
- Custom-built **Dense (FAISS IVF4096)** and **SPLADE** indexes over 528K documents

## Architecture

```
Fast & Shallow ──────────────────────────────────────────────► Slow & Deep

BM25  →  Dense  →  SPLADE  →  Bi-Encoder  →  Cross-Encoder  →  MonoT5  →  LLM
  │        │         │           │               │               │          │
keyword  semantic    both       filter         accurate        diverse   reasoning
```

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Stage 1: Hybrid Retrieval                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  │ BM25+RM3 │ │ BM25+RM3 │ │  Dense   │ │  Dense   │ │  SPLADE  │ │  SPLADE  │
│  │(Original)│ │  (Q2D)   │ │(Original)│ │  (Q2D)   │ │(Original)│ │  (Q2D)   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
│       └────────────┴────────────┴────────────┴────────────┴────────────┘
│                                      │
│                            Weighted RRF Fusion ──► Run 1 (1000 docs/query)
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                     Stage 2: Neural Cascade Reranking                       │
│                                                                             │
│   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐        │
│   │   Bi-Encoder    │ → │  Cross-Encoder   │ → │     MonoT5      │        │
│   │  bge-large-en   │   │   Ensemble (2x)  │   │  monot5-large   │        │
│   │   500 → 250     │   │    250 → 100     │   │   Final rank    │        │
│   └─────────────────┘   └──────────────────┘   └─────────────────┘        │
│                                      │                                     │
│                         MaxP Aggregation + Tiered Ranking ──► Run 2        │
└──────────────────────────────────────┼─────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────┐
│                      Stage 3: LLM Cascade Reranking                        │
│                                                                            │
│   ┌──────────────────────────┐   ┌──────────────────────────┐             │
│   │    GPT-4o-mini           │ → │       GPT-5              │             │
│   │  Top-30, sliding window  │   │  Top-20, single pass     │             │
│   └──────────────────────────┘   └──────────────────────────┘             │
│                                      │                                     │
│                   Weighted RRF (Run1 + Run2 + LLM) ──► Run 3               │
└────────────────────────────────────────────────────────────────────────────┘
```

## Results

### Training Set (50 queries, 301-350)

| Run | Method | MAP | NDCG | P@10 |
|-----|--------|-----|------|------|
| Run 1 | Hybrid Retrieval | 0.372 | 0.689 | 0.504 |
| Run 2 | + Neural Cascade | 0.381 | 0.672 | 0.498 |
| **Run 3** | **+ LLM Reranking** | **0.394** | **0.698** | **0.526** |

Each stage progressively refines the ranking, with the full pipeline (Run 3) achieving the best results across all metrics.

## Models Used

### Retrieval

| Method | Model | Purpose |
|--------|-------|---------|
| BM25+RM3 | Pyserini | Lexical matching + pseudo-relevance feedback |
| Dense | BAAI/bge-small-en-v1.5 | Semantic similarity (384-dim, FAISS IVF4096) |
| SPLADE | naver/splade-cocondenser-ensembledistil | Learned sparse retrieval with term expansion |
| Query2Doc | LLM-generated | Query expansion bridging vocabulary gap |

### Neural Reranking

| Stage | Model | Params | Role |
|-------|-------|--------|------|
| Bi-Encoder | BAAI/bge-large-en-v1.5 | 335M | Fast filtering (500 → 250 docs) |
| Cross-Encoder | bge-reranker-v2-m3 + ms-marco-MiniLM | 601M | Accurate scoring (250 → 100 docs) |
| MonoT5 | castorini/monot5-large-msmarco | 770M | Seq2seq relevance (final ranking) |

### LLM Reranking

| Stage | Model | Documents | Strategy |
|-------|-------|-----------|----------|
| Bulk | GPT-4o-mini | Top 30 | Sliding window (size=10, step=5) |
| Refine | GPT-5 | Top 20 | Single-pass precision refinement |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 recommended for neural models)
- Java 21 (required by Pyserini)
- OpenAI API key (for LLM reranking)

### Installation

```bash
git clone https://github.com/er1009/robust04-multi-stage-ranking.git
cd robust04-multi-stage-ranking

pip install -r requirements.txt

# Java (if not installed)
apt-get install openjdk-21-jdk-headless

# Set up API key
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Pre-Built Indexes (recommended)

We provide pre-built indexes on Google Drive so you can skip the indexing step entirely:

| Index | Model | Files | Size | Link |
|-------|-------|-------|------|------|
| **Dense** | BAAI/bge-small-en-v1.5 | `index.faiss`, `passages.pkl`, `config.json` | ~4.5 GB | [Download](https://drive.google.com/drive/folders/14-EPO-MXF_rQvFLcEMJZgq05sbInzJAg?usp=sharing) |
| **SPLADE** | naver/splade-cocondenser-ensembledistil | `splade_index.npz`, `passages.pkl`, `config.json` | ~3.1 GB | [Download](https://drive.google.com/drive/folders/1eNC2alOJ9nKHXBuyuikoYJtaMHiTqh7c?usp=sharing) |

Download both folders and pass their paths to `--dense-index-path` and `--splade-index-path`.

<details>
<summary><b>Or build indexes from scratch (~2 hours on GPU)</b></summary>

```bash
# Dense index (~45 min on GPU)
python -m src.dense_index \
    --index-path "path/to/dense_index" \
    --embedding-model "BAAI/bge-small-en-v1.5" \
    --chunk-size 1500 --chunk-overlap 200

# SPLADE index (~60 min on GPU)
python -m src.splade_index \
    --index-path "path/to/splade_index" \
    --chunk-size 1500 --chunk-overlap 200
```

</details>

### Run the Pipeline

```bash
# Training set (50 queries, with evaluation)
python -m src.main train \
    --output-dir results \
    --dense-index-path "path/to/dense_index" \
    --splade-index-path "path/to/splade_index" \
    --retrieval-k 2000 --rerank-depth 500

# Test set (199 queries)
python -m src.main test \
    --output-dir results \
    --dense-index-path "path/to/dense_index" \
    --splade-index-path "path/to/splade_index" \
    --retrieval-k 2000 --rerank-depth 500
```

## Project Structure

```
robust04-multi-stage-ranking/
├── src/
│   ├── main.py                 # CLI entry point and pipeline orchestration
│   ├── bm25_retrieval.py       # BM25 + RM3 retrieval via Pyserini
│   ├── dense_index.py          # FAISS dense index building
│   ├── splade_index.py         # SPLADE sparse index building
│   ├── document_processor.py   # Document chunking and contextual processing
│   ├── neural_reranker.py      # Bi-Encoder, Cross-Encoder, MonoT5 cascade
│   ├── llm_reranker.py         # LLM sliding-window listwise reranking
│   ├── fusion.py               # RRF and other fusion strategies
│   ├── few_shot_selector.py    # Dynamic few-shot example selection
│   ├── evaluation.py           # MAP, NDCG, P@k evaluation
│   ├── tune_retrieval.py       # BM25/RM3 parameter tuning
│   ├── tune_weights.py         # Fusion weight tuning
│   ├── tuning.py               # Tuning utilities
│   ├── data_loader.py          # Dataset loading (ir_datasets)
│   ├── trec_io.py              # TREC format I/O
│   ├── config.py               # Configuration and seeds
│   └── aggregation.py          # MaxP passage-to-document aggregation
├── data/
│   └── expanded_queries.csv    # Pre-computed Query2Doc expansions
├── run_pipeline.ipynb          # Google Colab notebook
├── requirements.txt
├── .env.example
└── README.md
```

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **6-way retrieval** | Lexical, dense, and sparse methods capture different relevance signals; Query2Doc bridges vocabulary gap |
| **Weighted RRF over score fusion** | Rank-based fusion is robust to incompatible score distributions across heterogeneous retrievers |
| **Cascade architecture** | Progressive filtering (fast → accurate) makes expensive models tractable over 1.7M passages |
| **Cross-encoder ensemble** | Two complementary models (bge-reranker + ms-marco-MiniLM) outperform either alone |
| **Tiered ranking** | Documents scored by the deepest cascade stage they reached, preserving stage-level ordering |
| **Contextual chunking** | Prepending document title to each passage preserves context lost during splitting |
| **Sliding-window listwise LLM** | Enables LLM comparison across documents despite context length limits |

## Configuration

### Retrieval Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BM25 k1 | 0.9 | Term frequency saturation |
| BM25 b | 0.4 | Document length normalization |
| RM3 fb_docs | 10 | Feedback documents for query expansion |
| RM3 fb_terms | 10 | Expansion terms added |
| RRF k | 60 | Rank fusion smoothing constant |
| Retrieval weights | BM25=1.0, Dense=0.3, SPLADE=1.0 | Per-retriever RRF weights |

### Cascade Cutoffs

| Stage | Input | Output |
|-------|-------|--------|
| Bi-Encoder | 500 docs | 250 docs |
| Cross-Encoder | 250 docs | 100 docs |
| MonoT5 | 100 docs | Ranked |
| GPT-4o-mini | 30 docs | 20 docs |
| GPT-5 | 20 docs | Final |

## References

- Cormack et al. (2009) -- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)
- Nogueira et al. (2020) -- [MonoT5: Document Ranking with T5](https://arxiv.org/abs/2003.06713)
- Sun et al. (2023) -- [RankGPT: LLMs as Reranking Agents](https://arxiv.org/abs/2304.09542)
- Formal et al. (2021) -- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- Wang et al. (2023) -- [Query2Doc: Query Expansion with LLMs](https://arxiv.org/abs/2303.07678)

## License

MIT
