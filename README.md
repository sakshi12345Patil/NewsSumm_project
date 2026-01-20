# NewsSumm Multi-Document Summarization

This repository contains implementations and experimental results for
multi-document news summarization on the NewsSumm dataset (Indian English).

The project includes multiple baseline models and a proposed novel model
for redundancy-aware long-context summarization.

This work is developed as part of a research internship and is intended
for journal and international conference submission.

## Dataset
The NewsSumm dataset consists of clusters of news articles describing the same event,
along with a human-written abstractive summary.

Dataset splits:
- train.jsonl
- val.jsonl
- test.jsonl

The dataset is not redistributed in this repository.
Please download it from the official source mentioned in the NewsSumm paper.
* NewsSumm dataset link (Zenodo): https://zenodo.org/records/17670865

## Implemented Models
**Baselines:**
- LexRank
- TextRank
- BART
- PEGASUS
- LED (Longformer Encoder-Decoder)

**Proposed Model:**
- SRLCSF (Structured Redundancy-Aware Long-Context Summarization Framework)

Ablation variants are included.

## Evaluation
Models are evaluated on the official test split using:
- ROUGE-1 (F1)
- ROUGE-2 (F1)
- ROUGE-L (F1)

Model-wise evaluation results are stored in the `results/` directory.

## Running an Example
```bash
python models/bart_baseline.py

## Repository Structure
data/       Dataset files (JSONL)
models/     Baselines and proposed model
results/    Evaluation outputs
docs/       Project documentation

##Environment
pip install -r requirements.txt

**Note:** The dataset files (`train.jsonl`, `val.jsonl`, `test.jsonl`, and `NewSumm Dataset.xlsx`) are **not included in this repository** due to their large size.  

To run the models and reproduce results, please download the NewsSumm dataset from the official source:
[NewsSumm dataset on Zenodo](https://zenodo.org/records/17670865)

Once downloaded, place the files in the `data/` folder of this repository.

## How to Run Baseline Models

Before running, make sure you have installed the required packages:

```bash
pip install -r requirements.txt

All baseline models can be run using their respective scripts:

Model	      Script Path	                            Example Command
BART	      models/bart_baseline.py	            python models/bart_baseline.py
LED	          models/led_baseline.py	            python models/led_baseline.py
LexRank	      models/baselines/lexrank_generate.py	python models/baselines/lexrank_generate.py
TextRank	  models/baselines/textrank_generate.py	python models/baselines/textrank_generate.py
PEGASUS	      models/baselines/pegasus_generate.py	python models/baselines/pegasus_generate.py
SRLCSF	      models/srlcsf/srlcsf_generate.py   	python models/srlcsf/srlcsf_generate.py


The generated summaries will be saved automatically in the results/ folder.
Evaluation (ROUGE scores) is computed automatically by each script.
Make sure the data/ folder contains the NewsSumm dataset (train.jsonl, val.jsonl, test.jsonl).
Tip: Run one model at a time to avoid high memory usage. Use CPU or GPU as available.
