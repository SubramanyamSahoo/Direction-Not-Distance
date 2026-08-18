# Mathematical object to executable implementation

Every central definition in the methods specification has a direct implementation.

| Definition or claim | Implementation | Runtime check or output |
|---|---|---|
| Pairwise response log probability | `aaai27_core.response_mean_logp` | chosen/rejected log probabilities in every alignment evaluation |
| Pairwise margin and softplus loss | `aaai27_core.pairwise_ranking_loss` | alignment training loss and full test margin |
| Full alignment-control gradient | `aaai27_core.full_alignment_gradient` | `control_alignment_loss_pre` and gradient-derived phase fields |
| Exact loss-bearing token budget | `aaai27_core.trim_labels_to_budget`, `train_shift_phase` | `phase_target_tokens_actual`; hard failure unless exact |
| Realized LoRA update | `aaai27_core.lora_vector`, `run_experiment.run_branch` | `raw_update_norm` |
| Global half-space projection | `aaai27_core.apply_update_control` | construction and stored first-order residuals |
| Coordinate mortality | `aaai27_core.apply_update_control` | killed fraction and positive-risk statistics |
| BF16 round-trip constraint check | `aaai27_core.apply_update_control` | `bf16_quantization_bound`, `constraint_tolerance`; hard failure on excess |
| Pairwise preference accuracy | `aaai27_core.evaluate_alignment` | full official test metrics at every phase |
| Held-out shift NLL and perplexity | `aaai27_core.evaluate_shift_nll` | phase-level plasticity metrics |
| One post-alignment drift anchor | `aaai27_core.drift_metrics` | raw and relative L2 drift |
| Deterministic behavioral sample size | `aaai27_core.wilson_required_n` | partition manifest and `verify_protocol.py` |
| Actual model generations | `aaai27_core.generate_behavior_sample` | checkpoint-specific JSONL files |
| Judge calibration | `score_behavior.wilson_interval` and `score_behavior.main` | `judge_validity.json`; fails closed unless lower bound exceeds chance |
| Exact paired sign-flip test | `analyze_results.exact_signflip_paired` | likelihood and judge exact-test CSVs |
| Holm correction | `analyze_results.holm_adjust` | family-adjusted p-values |
| Exact seed bootstrap | `analyze_results.exact_bootstrap_mean_ci` | group summary confidence intervals |
| Direction versus magnitude prediction | `analyze_results.direction_predictive_analysis` | leave-one-seed-out RMSE outputs |
| Hardware requirement | `verify_h100_pcie.py`, `aaai27_core.require_h100` | no CPU fallback; low-level CUDA and BF16 checks |
| Protocol arithmetic | `verify_protocol.py` | branch, seed, token, and sample-size report |
