# Benchmark and Analysis Notes

This document describes the benchmarking procedure and key observations for the NewsSumm project.

## Benchmarking Procedure
1. **Models included**
   - All 10 baseline models (LexRank, TextRank, BART, PEGASUS, LED, PRIMERA, LongT5 variants, etc.)
   - Proposed novel model (SRLCSF)

2. **Evaluation**
   - Run all models on the official test split
   - Metrics:
     - ROUGE-1, ROUGE-2, ROUGE-L (F1)
     - BERTScore (F1)
   - Record mean and 95% confidence intervals

3. **Result Organization**
   - Store outputs in `results/`
   - Prepare a summary CSV/Excel file:
     - Columns: `Model`, `Type (Enc-Dec / LLM / Hierarchical)`, `Context Length`, `Training Regime (FT / LoRA / Zero-shot)`, `ROUGE1`, `ROUGE2`, `ROUGEL`, `BERTScore`

## Error Analysis
1. Sample 100 clusters from the test set
2. Compare outputs of strongest baseline vs SRLCSF
3. Categorize errors:
   - Missing key events
   - Incorrect entities
   - Hallucinated content
   - Redundancy
   - Poor coherence

## Observations
- Baselines generally perform well on short clusters but struggle with redundancy
- SRLCSF shows improvement in:
  - Cross-article coherence
  - Factual consistency
  - Reduced repetition

## Plots and Tables
- Generate bar charts for ROUGE and BERTScore comparison
- Highlight best and second-best performance for each metric
- Include short narrative analysis for discussion section of paper or report
