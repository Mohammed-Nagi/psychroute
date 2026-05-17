# PsychRoute

**Fairness-constrained bandit routing for human–AI psychiatric triage.**

Reference implementation and unpublished paper. PsychRoute combines a fine-tuned MentalBERT classifier with a fairness-constrained LinUCB contextual bandit, routing each patient encounter to one of four governance-anchored actions: autonomous AI diagnosis, SHAP-scaffolded clinician review, expert escalation, or enriched-data refusal. Uncertainty is decomposed into epistemic and aleatoric components via Monte Carlo Dropout, and the action set is anchored to the Fabric governance taxonomy ([Jorgensen et al., 2025](https://arxiv.org/abs/2508.14119)).

> **Research code, not a clinical tool.** Demographic features in this repository are *synthetic proxies* drawn from WHO priors; the classifier is trained on self-reported Reddit text labelled by a third party, not on validated clinical instruments. PsychRoute exists to study the human–AI collaboration question — it must not be used to inform real medical decisions.

## Headline results

Bipolar Disorder vs Major Depressive Disorder, held-out test set, *N* = 3,626 encounters.

| Strategy | Diagnostic error | AI coverage | Escalation rate | Avg. cost / patient | Cumulative regret |
|---|---:|---:|---:|---:|---:|
| Model-Alone | 0.026 | 100.0 % | 0.0 % | 1.00 | 89 |
| Threshold | 0.023 | 99.3 % | 0.7 % | 1.05 | 78 |
| **PsychRoute** | **0.017** | 73.5 % | 26.5 % | 3.18 | **53** |

PsychRoute beats both the AI-only and human-only (*h*₀ = 0.93) baselines and achieves the lowest cumulative regret against a fixed-skill oracle. The full fairness audit, dual-variable trace, and reliability diagrams are in the paper.

## What's in this repo

```
psychroute/
├── README.md                          # this file
├── PsychRoute.ipynb                   # reference implementation, end to end
├── results/
│   ├── results.csv                    # headline metrics (Table 2 of the paper)
│   └── nutritional_label.csv          # disclosure card (Appendix E)
└── PsychRoute.pdf                     # paper
```

The fine-tuned classifier weights (`mentalbert_bd_mdd.pt`, ≈ 440 MB) are not tracked in git — the notebook regenerates them on first run.

## Setup

### Dependencies

```bash
pip install torch transformers huggingface_hub datasets kagglehub \
            shap scikit-learn pandas matplotlib seaborn tqdm fairlearn scipy
```

Tested with Python 3.11 and PyTorch 2.x.

### Credentials

Set these environment variables **before** opening the notebook:

```bash
export HF_TOKEN="hf_..."                  # HuggingFace, read-only token is fine
export KAGGLE_USERNAME="your_username"    # optional if ~/.kaggle/kaggle.json exists
export KAGGLE_KEY="your_kaggle_key"       # optional if ~/.kaggle/kaggle.json exists
```

The HuggingFace token gates the `mental/mental-bert-base-uncased` checkpoint; the Kaggle credentials gate the [Mental Disorder Classification dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). The notebook raises a clear error if `HF_TOKEN` is missing — it will never run with a hard-coded token.

### Run

```bash
jupyter notebook PsychRoute.ipynb
```

Run all cells in order. Expected wall time on a single T4 or A10 GPU is ≈ 25 minutes for fine-tuning plus ≈ 5 minutes for the routing simulation and figures. CPU-only runs work but take several hours.

## Reproducibility

The notebook seeds `numpy`, `torch`, and the simulator's `default_rng` with `SEED = 42`. CPU-only runs should be bit-for-bit reproducible; GPU runs introduce small cuBLAS / cuDNN non-determinism that does not affect headline numbers.

The regret oracle uses its own `np.random.default_rng(SEED + 1)` stream rather than the in-loop `rng`, which keeps the regret computation independent of the bandit's stochastic decisions and matches the values reported in Table 2.

## Citation

```bibtex
@inproceedings{nagi2026psychroute,
  author    = {Nagi, Mohammed},
  title     = {{PsychRoute}: Fairness-Constrained Bandit Routing for
               Human-{AI} Psychiatric Triage},
  booktitle = {AIMS 2026 Workshop, Algorithms for Human-AI Collaboration},
  year      = {2026},
  address   = {Cape Town, South Africa},
}
```

## Acknowledgments

This work was completed as the final project for the *Responsible AI* course at the [African Institute for Mathematical Sciences (AIMS)](https://aims.ac.za), Cape Town, taught by Prof. Umang Bhatt. The MentalBERT checkpoint is due to [Ji et al. (2022)](https://aclanthology.org/2022.lrec-1.778/); the dataset is due to [Sarkar (2023)](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health).
