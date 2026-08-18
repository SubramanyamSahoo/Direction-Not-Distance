# Reviewer criticism to implementation change

## The original method lacked a principled mechanism

**Change:** replace the apoptosis metaphor, learned death rate, Gaussian noise, Gumbel interval, and broken FOMAML path with two closed-form controls derived from a first-order alignment-loss constraint. Global projection is the minimum Euclidean correction; coordinate mortality deletes exactly the locally harmful coordinate contributions.

## It was unclear whether the model learned the new distribution

**Change:** every phase reports held-out shift NLL and perplexity alongside alignment. The principal result is a stability-plasticity frontier, not alignment retention in isolation.

## Preference Score was difficult to interpret

**Change:** the primary metric is pairwise preference accuracy on the official test split. Mean and median response-conditional margins are also reported.

## The external reward model produced a constant score

**Change:** the judge scores actual generated responses, not fixed human reference text. It is calibrated on chosen/rejected labels first. The judge is accepted only when the Wilson lower confidence bound for chosen-over-rejected accuracy is strictly above chance. Constant and non-finite outputs fail closed, and input-at-limit incidence is reported.

## Only 50 probes were used

**Change:** likelihood-based alignment evaluation uses the complete official test split. Behavioral sample size is calculated from an explicit Wilson-interval precision target.

## Three seeds were underpowered

**Change:** the primary Qwen3-8B experiment uses seven paired seeds, the minimum needed for an exact two-sided sign-flip test to attain Holm-adjusted p < .05 across all three pairwise method comparisons. Qwen3-14B is explicitly designated as a three-seed scale replication, so its inferential limits are not concealed.

## The shift was unusually benign

**Change:** compare benign OpenWebText adaptation with a conflict shift trained toward rejected harmlessness responses. Alignment training, intervention control, conflict training, conflict evaluation, and the official alignment test are all disjoint.

## The claim that LoRA is intrinsically durable was too broad

**Change:** test conditional durability across shift regimes. No unconditional durability claim is encoded into the analysis.

## Parameter distance and alignment were confused

**Change:** use one anchor, the post-alignment LoRA checkpoint, for every cumulative drift measurement. Compare distance with update direction directly.

## The original mortality implementation did not match the paper

**Change:** remove it. The new coordinate mortality includes every trainable LoRA tensor, has no magnitude top-k error, no noise, and no unsupported differentiability claim.

## Half-life fitting imposed an inappropriate exponential model

**Change:** remove exponential half-life as the primary result. Report phase trajectories, exact token exposure, and stability-plasticity outcomes without assuming monotonic decay.

## Reproducibility and crashes

**Change:** pin Python research packages, record exact Hub commit SHAs, save exact partition IDs/hashes, checkpoint every phase, and resume from completed phase boundaries.
