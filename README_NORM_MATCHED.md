# AAAI-27 targeted follow-up: Norm-Matched shrinkage control

## Why this experiment exists

The completed primary experiment shows that Coordinate Mortality (CM) improves held-out preference behavior, but CM also reduces the update norm. A reviewer can therefore argue that the effect comes from generic shrinkage rather than coordinate selection.

This follow-up tests that alternative explanation directly.

At each phase, with alignment-control gradient `g` and raw continual-learning update `u`, compute the counterfactual CM update:

```text
v_CM,i = 0    if g_i u_i > 0
         u_i  otherwise
```

Then derive

```text
r = ||v_CM||_2 / ||u||_2
```

and apply the Norm-Matched control

```text
v_NM = r u.
```

No intervention strength is tuned. `r` is determined by the same phase, model state, gradient, and raw update. The new update has the same FP32 norm as CM would have had at that exact state, but keeps the original update direction and deletes no coordinates.

**Primary follow-up hypothesis:** Coordinate Mortality yields higher final held-out preference accuracy than Norm-Matched shrinkage. The two regimes (benign and conflict) form the primary family and are Holm-corrected across the two exact paired tests.

Secondary outcomes: preference margin, shift NLL improvement (plasticity), relative L2 drift, and ArmoRM behavioral scores.

## Scope

- Qwen3-8B only.
- Seeds 42–48 (same seven primary seeds).
- Benign + conflict shift.
- Six phases.
- Exactly 819,200 loss-bearing target tokens per phase.
- Same model revision, partitions, optimizer, LoRA configuration, data order, and phase-specific stochastic streams as the completed primary study.
- Existing preference-tuned checkpoints are loaded from `pci_h100_outputs`; the alignment stage is **not** retrained.

This is 14 new branches = 84 trained phases, not a rerun of the full study.

## Files

- `run_norm_matched.py` — checkpointed/resumable H100 training + final generations.
- `score_norm_matched_behavior.py` — ArmoRM scoring with the existing compatibility shim.
- `analyze_norm_matched.py` — exact paired tests, summaries, figures, numerical norm-match verification.
- `analyze_hidden_cancellation.py` — zero-training analysis explaining why global projection can miss coordinate-level conflict.
- `verify_norm_matched_math.py` — deterministic mathematical unit check.
- `run_norm_matched.sh`, `run_norm_matched_judge.sh`, `run_norm_matched_analysis.sh` — launchers.

## Expected runtime

Based on the completed 8B run, approximately 24–30 H100 PCIe hours. Conflict phases dominate runtime. The scripts are resumable at phase boundaries.
