# Claim guardrails for the AAAI paper

Do not write any of the following before the complete results exist:

- direction predicts alignment loss better than magnitude;
- coordinate mortality outperforms global projection;
- either control preserves alignment without a plasticity cost;
- LoRA alignment is intrinsically durable;
- conflict shift always destroys alignment;
- the external judge validates every likelihood-based conclusion.

Permitted design-level statements before results:

- the two controls satisfy their FP32 first-order constraints by construction;
- training and control partitions are disjoint from the official test split;
- phase target-token budgets are exact;
- the external judge is tested against known chosen/rejected labels;
- seven paired seeds on the primary 8B scale permit the stated exact-test resolution after Holm correction;
- the three-seed 14B run is scale-replication evidence, not a second confirmatory test.
