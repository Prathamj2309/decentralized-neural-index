# Semantic Metadata Protocol for Hybrid Search

## Overview

This project explores a decentralized search architecture in which web pages publish quantized semantic embeddings as metadata. These embeddings enable efficient hybrid retrieval by combining lexical ranking (BM25) with semantic similarity scoring.

The goal is to evaluate whether publisher-provided embeddings can improve retrieval quality while reducing query-time computation.

## Architecture

The system consists of:

1. Offline Index Builder
   - Computes normalized sentence embeddings for documents
   - Stores both float32 and int8-quantized versions

2. Lexical Index
   - BM25 inverted index over the document corpus

3. Hybrid Retrieval
   - Step 1: BM25 retrieves top-k candidates
   - Step 2: Query embedding is computed
   - Step 3: Precomputed document embeddings are used for reranking
   - Final score is a weighted combination of lexical and semantic scores

## Experiments

We evaluate:

- BM25 baseline
- Hybrid (float32 embeddings)
- Hybrid (int8 quantized embeddings)

Metrics include:
- Query latency
- Storage footprint
- Ranking quality (planned NDCG/MRR)

## Key Findings (Preliminary)

- Precomputed embeddings eliminate document encoding at query time
- Int8 quantization reduces storage by ~4x
- Semantic reranking improves relevance for ambiguous or synonym-heavy queries

## Future Work

- Formal evaluation using ranking metrics
- Adversarial embedding analysis
- Protocol standardization for web metadata publishing
