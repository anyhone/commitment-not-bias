# Monitor-gated vs always-on activation steering for fairness

Code to reproduce an independent evaluation of two activation-steering designs for social bias in LLMs:

- **CAA** (always-on): a steering vector is added at every position after the instruction.
- **FairSteer** (monitor-gated): a linear probe reads the last prompt token and the vector is added at that position only when the probe flags the activation as biased.

Models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 and Qwen3.5-9B, all in 4-bit NF4.
Benchmarks: BBQ, StereoSet and MMLU.

The repository contains everything needed to rerun the experiments, plus the small result files of our runs (`results/*/report`, `results/*/diag`, probe reports) as a reference to compare against.

## Layout

```
configs/              one yaml per model (HF id, pinned revision, chat-template options, batch sizes, layers)
steering/             library: data loading and splits, model loading, steering hooks, scoring, metrics, probe
scripts/              pipeline steps, test evaluation, diagnosis, post-hoc analyses
tests/                test_data.py (CPU), test_gpu.py (GPU regression tests)
data/splits.json      train/dev/test split by BBQ template and StereoSet target (versioned; the datasets are downloaded)
results/<model>/      report/ (sweep, selection, test results, post-hoc analyses), diag/, probes*/report.json
```

Large outputs are not versioned: per-item predictions (`results/<model>/eval/`), baseline activations (`results/<model>/baseline/`), steering vectors (`vectors.pt`), probe weights (`probes*/probe_L*.pt`) and logs. The pipeline below regenerates them.

## Setup

Requirements: Python 3.10, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with at least 16 GB and a CUDA 12.x driver. The experiments ran on a single NVIDIA RTX 5000 (16 GB, Turing). Llama-3.1 is a gated model on the Hugging Face Hub: accept its licence and log in with `huggingface-cli login` before running.

```bash
uv sync                                   # torch 2.7.1 from the PyTorch cu126 index, the rest from PyPI (uv.lock)
uv run python scripts/download_data.py    # BBQ (checked against SHA-256 hashes), StereoSet, MMLU -> data/
uv run python -m pytest tests/test_data.py -q
```

The model revisions are pinned in `configs/*.yaml`.

## Rerunning the experiments

Each step is idempotent or resumes where it stopped. Long runs are meant for `tmux`.

```bash
M=llama                                   # or mistral_instruct, qwen
scripts/run_pipeline.sh $M                # baseline, vectors, probes, GPU tests, dev sweep, report (selection on dev)
scripts/run_test.sh $M                    # final test evaluation of the selected configurations, run once
scripts/run_diag.sh                       # last-token diagnosis (all models)
scripts/run_posthoc.sh                    # runs for the post-hoc monitor analysis (FairSteer without monitor, CAA at FairSteer's configuration)
uv run python scripts/direct_comparisons.py llama mistral_instruct qwen
uv run python scripts/posthoc_gates.py    # simulated gates, 2x2 position x monitor, primary hypotheses
uv run python scripts/posthoc_mechanism.py
```

Approximate GPU time on the RTX 5000: one test condition takes about 30 min (Llama), 26 min (Mistral) and 84 min (Qwen); one dev condition about 3, 2.6 and 9.4 min. The dev sweep has about 160 conditions per model.

## Protocol

- **Splits.** BBQ is split by template and StereoSet by target (30% train, 20% dev, 50% test, `data/splits.json`). Steering vectors and probes use the train split only.
- **Selection.** Layer and strength are chosen on dev with the same criterion for both methods (`scripts/report.py`). BBQ: lowest |s_amb| with a drop of at most 2 points in disambiguated accuracy and 1 point in MMLU. StereoSet: lowest |SS − 50| with a drop of at most 2 points in LMS and 1 point in MMLU.
- **Test.** The selected configurations are evaluated once on test, with 95% paired-bootstrap CIs clustered by BBQ template / StereoSet target.
- **Post-hoc analyses.** The runs without the monitor (`--threshold -1`) and the analyses in `posthoc_gates.py` and `posthoc_mechanism.py` were added after the test evaluation, at the configurations already selected. Their protocol was written before running them; no hyperparameter was reselected.

## Naming

- `probes/`: BAD probe trained with the labels of the official FairSteer code (**code specification**: biased = the model gives the stereotyped answer and it is wrong; unbiased = the answer is correct). This is the default (`--labels code`).
- `probes_text/`: probe trained with a reading of Sec. 3.1 of the FairSteer paper (`--labels text`) (**text specification**: biased = stereotyped answer; unbiased = "unknown"). Used as an ablation.
- Condition tags: `<method>_L<layer>_<rel|raw><coef>[_t<threshold>][_cat][_probes_text]`. `rel` is a coefficient relative to the mean residual-stream norm; `raw` multiplies the unnormalised vector. `_t-1` is FairSteer without the monitor.

## Differences from the original FairSteer setup

| | FairSteer (Li et al., 2025) | This repository |
|---|---|---|
| Layer choice | highest probe accuracy | by effect on dev, same criterion as CAA |
| Classifier | cuML logistic regression | scikit-learn logistic regression + StandardScaler, balanced class weights |
| Probe labels | code: wrong stereotyped vs correct answer | same (code specification); text specification as an ablation |
| Strength | α = 1 on the raw vector | relative coefficient grid, plus α = 1 raw as a reference |
| Vector pairs | 110 ambiguous items (10 per category, 11 categories) | all ambiguous train items (4,592, 9 categories) |
| BBQ items | 58,492 (with intersectional categories) | 31,372 (9 single-attribute categories) |
| Other benchmarks | CrowS-Pairs, UNQOVER, CEB | StereoSet |
| MMLU | 5-shot | 0-shot |
| Precision | full | 4-bit NF4 |

Minor differences in the probe data: the official code trains on all of BBQ with zero-shot and 8-shot prompts and a random 80/20 item split; here the probe uses the train split only, zero-shot in the chat template, MMLU neutral examples from `auxiliary_train`, plus StereoSet examples. The 16 BBQ items without a stereotyped target are excluded. The position and the gate (probe on the last prompt token at layer ℓ, steering at that position only) are the same as in the official code.

## Data and models

The datasets ([BBQ](https://github.com/nyu-mll/BBQ), [StereoSet](https://huggingface.co/datasets/McGill-NLP/stereoset), [MMLU](https://huggingface.co/datasets/cais/mmlu)) and the model weights are not redistributed here. They are downloaded from their sources and remain under their own licences.

## Licence

Code: MIT (see `LICENSE`).
