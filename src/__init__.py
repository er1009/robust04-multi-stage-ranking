"""Multi-stage document ranking pipeline for TREC ROBUST04.

Combines lexical (BM25+RM3), dense (FAISS), and learned sparse (SPLADE)
retrieval with neural cascade reranking (Bi-Encoder → Cross-Encoder → MonoT5)
and LLM-based reranking (GPT-4o-mini → GPT-5).
"""
