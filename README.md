# Structured Inference with Prompt-Based LLMs

This repository contains the code, prompts, data, and figures for the manuscript **Structured Inference with Prompt-Based LLMs: A Theory-Driven Framework for Identifying Narratives of Ontological Security**.

## Repository structure

```text
data/
  Data_Train.xlsx
  Data_Test.xlsx

prompts/
  direct_prompting.txt
  baseline_prompting.txt
  conceptual_decomposition.txt
  self_reflective_validation.txt
  retrieval_augmented_grounding.txt
  few_shot_prompting.txt

scripts/
  baseline_prompting.py
  conceptual_decomposition.py
  self_reflective_validation.py
  retrieval_augmented_grounding.py
  few_shot_prompting.py
  build_oskr_index.py
  nrc_emotion_lexicon.py
  svm.py
  knn.py
  naive_bayes.py
  random_forest.py
  mlp.py
  bert_train.py
  bert_predict.py

figures/
  combined_prompting_methods.pdf
  confusion_matrix_4methods.pdf
  confusion_matrix_4methods_row_normalized.pdf
  confusion_matrix_few_shot.pdf
  kappa_across_confidence_threshold.pdf
  Kappa_CI_Pvalues.pdf
  kappa_vs_temperature_llama_70b.pdf
  query_different.pdf
  runtime_costs.pdf
  self_reflective_confidence_barplot.pdf
  topK_chunk_ablation.pdf
  trump_monthly_speeches.pdf
  Trump_Narrative_Distribution.pdf
  trump_speech_length.pdf
  two_baseline_F1.pdf
```

## Setup

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

For scripts that call LLM APIs, keep API keys outside the repository. A local `.env` file can be created from `.env.example`, but `.env` should not be committed to GitHub.

## Running the scripts

Scripts can be run individually from the repository root. For example:

```bash
python scripts/baseline_prompting.py
python scripts/conceptual_decomposition.py
python scripts/self_reflective_validation.py
python scripts/retrieval_augmented_grounding.py
```

Traditional machine-learning baselines can be run with:

```bash
python scripts/svm.py
python scripts/knn.py
python scripts/naive_bayes.py
python scripts/random_forest.py
python scripts/mlp.py
python scripts/nrc_emotion_lexicon.py
```

The BERT baseline is separated into training and prediction scripts:

```bash
python scripts/bert_train.py
python scripts/bert_predict.py
```

The OSKR index can be built with:

```bash
python scripts/build_oskr_index.py
```

## Data

`Data_Train.xlsx` contains the training data used for supervised baselines.  
`Data_Test.xlsx` contains the test data used for evaluation.

## Prompts

The prompt templates used in the experiments are stored in the `prompts/` folder. They correspond to the prompting methods reported in the manuscript.

## Notes

- Do not upload API keys, local `.env` files, private paths, or temporary cache files.
- LLM-based results may depend on the exact model endpoint and API configuration used at runtime.
- The PDF files in `figures/` are included to document the figures reported in the manuscript and supplementary materials.
