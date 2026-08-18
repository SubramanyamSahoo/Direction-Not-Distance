# Methods specification: directional update control under continual alignment shift

## Central question

When a preference-aligned language model is continually adapted, does alignment loss depend more on the **direction** of the realized update than on its Euclidean magnitude, and can parameter-free update controls preserve alignment without eliminating plasticity?

## Models and adaptation

The study uses `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B` in BF16. The base weights are frozen. LoRA rank, alpha, dropout, target modules, learning rate, weight decay, sequence length, phase budget, phase count, warmup ratio, and gradient clipping are inherited from the earlier CoLLAs protocol so that the new study changes the scientific question rather than silently retuning the system.

## Alignment objective

For a prompt history `x`, human-preferred response `y+`, and rejected response `y-`, define the token-normalized response log probability

```text
q_theta(y | x) = (1 / |y|) sum_t log p_theta(y_t | x, y_<t).
```

The preference margin and training loss are

```text
m_theta = q_theta(y+ | x) - q_theta(y- | x)
L_A(theta) = E softplus(-m_theta).
```

This is a direct pairwise preference objective. It is not described as RLHF and introduces no DPO temperature.

## Data partitions

The HH-RLHF `harmless-base` training split is deterministically shuffled once. Three disjoint partitions are then formed:

1. alignment training;
2. alignment control, used only to calculate intervention gradients;
3. conflict-shift evaluation, held out from every optimization step;
4. conflict-shift training, using rejected responses from the remaining records.

The complete official harmless-base test split is never used for training, intervention control, or shift-NLL evaluation. It is reserved for the primary pairwise alignment evaluation and the deterministic behavioral sample. OpenWebText is independently split into train and held-out evaluation subsets. Exact Hub commit SHAs and partition IDs are written to manifests.

## Shift regimes

### Benign shift

Continual causal-language-model training on OpenWebText.

### Conflict shift

Continual response-only training toward rejected HH-RLHF responses from the disjoint conflict partition.

The contrast is deliberately controlled: benign adaptation has weak semantic overlap with the preference objective, while conflict adaptation directly points toward behavior rejected by preference annotators.

## Phase budget

Each phase contains exactly `819,200` loss-bearing target tokens. The final batch is deterministically masked so the phase neither overshoots nor undershoots this budget. Six phases are retained from the earlier protocol.

## Directional quantities

At the beginning of phase `t`, calculate the exact mean alignment-control gradient over the full control partition:

```text
g_t = grad_theta L_A(theta_t).
```

After ordinary shift training, let

```text
u_t = theta_raw - theta_t
```

be the realized LoRA update. First-order alignment-loss change is

```text
g_t^T u_t.
```

The code records the raw update norm, cosine with the alignment-descent direction `-g_t`, positive coordinate-risk fraction, and cumulative distance from the post-alignment anchor.

## Conditions

### 1. Unconstrained

```text
v = u.
```

### 2. Global projection

Solve the Euclidean projection problem

```text
min_v ||v - u||_2^2  subject to  g^T v <= 0.
```

Its closed-form solution is

```text
v = u                                      if g^T u <= 0
v = u - (g^T u / ||g||_2^2) g             otherwise.
```

This is the minimum-change dense correction satisfying the local first-order constraint.

### 3. Coordinate mortality

```text
v_i = 0       if g_i u_i > 0
v_i = u_i     otherwise.
```

Therefore

```text
g^T v = sum_{i: g_i u_i <= 0} g_i u_i <= 0.
```

No death rate, top-k threshold, noise scale, interval distribution, Gumbel variable, or meta-learning loop is introduced. Intervention strength is measured, not configured.

## Numerical guarantee

The controls are constructed in FP32 and written to BF16 LoRA tensors. The code re-reads the stored state and records the actual first-order residual. It computes a BF16 rounding-error bound from unit roundoff and the vectors themselves. A controlled update that exceeds this computed bound causes a hard failure.

## Primary alignment evaluation

On the complete official test split, report:

- pairwise preference accuracy;
- mean and median token-normalized preference margin;
- chosen and rejected response log probability;
- coverage and truncation diagnostics.

## Plasticity evaluation

At every phase, report held-out shift negative log likelihood and perplexity. Alignment retention is therefore interpreted jointly with successful learning of the new distribution.

## Independent behavioral evaluation

Every base, aligned, and final checkpoint generates responses to the same deterministic held-out prompt sample. Sample size is calculated from a preregistered Wilson-interval precision target rather than inserted as a hidden probe count. ArmoRM scores actual generated responses as well as the paired chosen and rejected references.

Before generated-response scores are used, judge validity is measured by:

- chosen-over-rejected accuracy;
- a Wilson confidence interval for that accuracy;
- tie fraction;
- score variance;
- maximum-length incidence;
- constant-output detection.

The judge passes only if the Wilson lower confidence bound for chosen-over-rejected accuracy is strictly above chance. Constant or non-finite output fails closed. Independent-judge method comparisons use the same paired exact testing and Holm correction as the likelihood-based endpoints.

## Statistical design

Qwen3-8B is the primary inferential scale and uses seven paired seeds. Seven is the smallest number for which a two-sided exact sign-flip test can attain Holm-adjusted `p < .05` across the three pairwise method comparisons:

```text
3 * (2 / 2^7) = 0.046875.
```

Qwen3-14B uses three paired seeds as a predeclared scale replication to keep the project computationally bounded. Its effect estimates and intervals are reported, but its hypothesis tests are explicitly labelled descriptive because their exact p-value resolution cannot support the primary confirmatory claim.

Final method comparisons use exact paired sign-flip randomization tests and Holm correction within each model-regime-metric family. Group confidence intervals enumerate every ordered seed-level bootstrap resample, avoiding Monte Carlo bootstrap noise. Choosing seven seeds controls test resolution; it is not represented as a conventional variance-based power calculation.

## Predictive analysis

On the unconstrained condition, leave-one-seed-out linear prediction compares:

1. update-direction cosine;
2. update magnitude.

The target is observed phase-level preference-margin change. Per-seed RMSE differences are compared with the same exact paired sign-flip test.

## Claim discipline

The implementation tests, but does not presuppose, that direction predicts alignment change better than magnitude or that either intervention improves the stability-plasticity frontier. Manuscript claims must be written only after inspecting the generated results.
