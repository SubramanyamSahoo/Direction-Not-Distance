# Direction, Not Distance?

Research code and results for studying alignment-relevant preference drift during continual learning.

## Overview

We study whether the geometry and coordinate structure of continual-learning updates can help control preference drift in language models.

The experiments use Qwen3-8B as the primary model with seven paired seeds and Qwen3-14B as a scale replication. We compare continual learning under:

- benign OpenWebText adaptation
- alignment-conflicting training toward rejected HH-RLHF responses

The main update conditions are:

- **Unconstrained**
- **Global Projection**
- **Coordinate Mortality**
- **Norm-Matched Shrinkage** as a targeted follow-up control

Coordinate Mortality suppresses update coordinates satisfying

`g_i * u_i > 0`

where `g` is a held-out preference-control gradient and `u` is the realized continual-learning update.

## Main findings

Across the primary Qwen3-8B experiments, Coordinate Mortality improves held-out pairwise preference accuracy relative to both unconstrained learning and Global Projection.

A targeted norm-matched control shows that this effect is not explained by generic update shrinkage: Coordinate Mortality outperforms Norm-Matched Shrinkage under both benign and conflicting continual learning.

The hidden-cancellation analysis also shows that substantial coordinate-level alignment conflict can be obscured in the global inner product because positive and negative coordinate contributions cancel.

Independent ArmoRM behavioral evaluation is included as a secondary evaluator and does not uniformly reproduce the likelihood-based preference gains.

## Repository structure

- `run_experiment.py` — main continual-learning experiment
- `aaai27_core.py` — shared experimental utilities
- `run_norm_matched.py` — targeted Norm-Matched follow-up
- `analyze_results.py` — primary statistical analysis
- `analyze_norm_matched.py` — targeted follow-up analysis
- `analyze_hidden_cancellation.py` — coordinate-cancellation analysis
- `score_behavior.py` — behavioral reward-model evaluation
- `score_norm_matched_behavior.py` — follow-up behavioral evaluation
- `results/main/` — primary experiment results and figures
- `results/norm_matched/` — targeted follow-up results and figures
- `docs/` — methodology and experimental protocol documentation

## Models and data

Experiments use:

- Qwen/Qwen3-8B
- Qwen/Qwen3-14B
- Anthropic/hh-rlhf
- Elriggs/openwebtext-100k
- RLHFlow/ArmoRM-Llama3-8B-v0.1

Model weights and large training checkpoints are intentionally not stored in this repository.

## Reproducibility

The repository contains the experiment scripts, frozen model/data revisions, deterministic dataset partitions, statistical analysis code, final result tables, and generated figures.

Large model weights, LoRA checkpoints, and intermediate generation artifacts are excluded from Git because they can be regenerated from the provided code and pinned revisions.

## Status

Research repository. Results should be interpreted together with the reported statistical tests and evaluation-validity limitations.
