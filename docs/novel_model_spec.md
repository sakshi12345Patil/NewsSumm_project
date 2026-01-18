# Novel Model Specification

This document describes the proposed novel model for the NewsSumm multi-document summarization project.

## Model Name
**SRLCSF** — Structured Redundancy-Aware Long-Context Summarization Framework

## Motivation
- Existing baselines (BART, PEGASUS, LED, etc.) struggle with redundancy and cross-article synthesis in multi-document clusters.
- SRLCSF is designed to:
  - Model sentence- and document-level relationships
  - Reduce redundancy across cluster articles
  - Produce coherent, factual summaries

## Architecture
1. **Encoder**
   - Hierarchical encoding:
     - Sentence-level embeddings for each article
     - Document-level embeddings aggregated from sentences
     - Cluster-level representation combining all articles

2. **Planner / Graph Layer**
   - Captures cross-article entity and event relationships
   - Determines which sentences or content are salient for final summary

3. **Decoder**
   - Global decoder conditioned on cluster-level plan
   - Generates abstractive summary using cross-attention to encoded cluster representation

4. **Loss Functions**
   - Main cross-entropy loss for summary generation
   - Optional auxiliary loss for salience prediction
   - Coverage penalty to reduce repeated content

## Implementation
- Built on top of a baseline model (e.g., LongT5 or LED)
- Additional modules:
  - Graph aggregator for cross-article relationships
  - Planning head for sentence importance

## Training
- Fine-tune from baseline checkpoint on NewsSumm
- Hyperparameters:
  - Max input tokens: 4096–8192 (model dependent)
  - Max target tokens: 128–256
  - Batch size, learning rate, warmup steps tuned for dataset
- Evaluation performed on validation set using ROUGE metrics

## Deliverables
- `models/srlcsf_model.py` — model architecture
- `models/srlcsf_generate.py` — inference and summary generation
- Experimental results stored in `results/` directory
