# Frozen targeted follow-up hypothesis (before Norm-Matched training)

## Motivation

Coordinate Mortality (CM) both selects coordinates using the sign of `g_i u_i` and reduces the update norm. The completed primary experiment therefore does not by itself identify whether CM's preference gains arise from *which coordinates are removed* or simply from *making the update smaller*.

## Targeted control

For every phase, at the current Norm-Matched branch state, calculate the same held-out alignment-control gradient `g` and the raw continual-learning update `u`. Construct the counterfactual CM update `v_CM` by zeroing coordinates with `g_i u_i > 0`. Derive

`r = ||v_CM||_2 / ||u||_2`

and apply

`v_NM = r u`.

`r` is computed, not tuned. Norm-Matched shrinkage therefore preserves the raw update direction while matching the L2 norm of the counterfactual CM update at the same state.

## Primary targeted hypothesis

On Qwen3-8B, Coordinate Mortality has higher final held-out pairwise preference accuracy than Norm-Matched shrinkage.

The primary family contains exactly two paired exact sign-flip tests: one for benign shift and one for conflict shift, both across seeds 42--48. Holm correction is applied across those two p-values.

## Secondary outcomes

- mean preference margin;
- held-out shift NLL improvement (plasticity);
- relative L2 drift;
- independent ArmoRM behavioral metrics;
- numerical verification that the stored Norm-Matched update matches the counterfactual CM norm within the computed BF16 rounding bound.

These secondary outcomes are interpreted as supporting/exploratory and are not substituted for the primary endpoint after results are observed.

## Interpretation rule

- If CM > NM on the primary endpoint, generic norm shrinkage is insufficient to explain CM; coordinate selection carries additional information.
- If CM ~= NM, the original CM gain may be largely attributable to update shrinkage.
- If the result differs by regime, the paper reports the regime dependence rather than claiming a universal mechanism.
