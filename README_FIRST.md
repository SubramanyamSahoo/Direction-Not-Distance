# AAAI-27 H100 PCIe directional-control project

This bundle is designed for **one NVIDIA H100 PCIe 80 GB**. It does not use Fabric Manager, NVSwitch, distributed training, or CPU fallback.

## 1. Verify the new instance before uploading or installing anything

Open a terminal on the new PCIe instance and run:

```bash
python -c 'import ctypes; print("cuInit:", ctypes.CDLL("libcuda.so.1").cuInit(0))'
```

Continue only if it prints:

```text
cuInit: 0
```

Then run:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

Continue only if CUDA is `True` and the GPU name contains `H100`.

## 2. Upload and extract

```bash
unzip AAAI27_H100_PCIe_Directional_Control.zip
cd AAAI27_H100_PCIe_Directional_Control
```

## 3. Create the isolated environment

The setup preserves the CUDA-enabled PyTorch supplied by Lambda and installs the pinned research stack into `.venv`.

```bash
bash setup_pcie_env.sh
```

Required final lines:

```text
H100 PCIE PREFLIGHT PASSED
METHOD MATH CHECK PASSED
SETUP COMPLETE
```

## 4. Authenticate to Hugging Face

```bash
source .venv/bin/activate
hf auth login
```

When asked `Add token as git credential?`, enter `n`. A read token is enough.

## 5. Cache exact model and dataset revisions

```bash
python -u prepare_assets.py 2>&1 | tee prepare_assets.log
```

Required final line:

```text
ASSET PREPARATION COMPLETE
```

This writes `asset_manifest.json` with exact Hub commit SHAs.

## 6. Run the primary 8B experiment in tmux

The primary inferential experiment uses seven paired seeds. This is the only
scale on which Holm-adjusted exact method-comparison claims are confirmatory.

```bash
tmux new -s aaai27_8b
bash run_train_8b.sh
```

Detach with `Ctrl+B`, release, then press `D`. The job continues if your browser or local internet disconnects.

Reconnect:

```bash
tmux attach -t aaai27_8b
```

Monitor from another terminal:

```bash
tail -f pci_h100_outputs/logs/train_8b.log
```

If the Python process stops after a completed phase, run the same command again. The runner resumes from the latest complete phase.

## 7. Run the 14B scale replication

The 14B run uses three paired seeds as a deliberately smaller scale
replication. Treat its intervals and exact-test p-values as descriptive
replication evidence, not as a second fully powered confirmatory study.

After 8B completes:

```bash
tmux new -s aaai27_14b
bash run_train_14b.sh
```

Detach in the same way.

To run both sequentially in one tmux session instead:

```bash
tmux new -s aaai27_all
bash run_train_all.sh
```

## 8. Run the independent behavioral judge

After both model runs finish:

```bash
tmux new -s aaai27_judge
bash run_judge.sh
```

The judge first measures chosen-over-rejected accuracy, its Wilson confidence interval, score variance, ties, and context-limit incidence. It proceeds only when the Wilson lower bound is strictly above chance; constant and non-finite outputs fail closed.

## 9. Produce tables and figures

```bash
bash run_analyze.sh
```

## 10. Important outputs

Under `pci_h100_outputs/results/`:

- `alignment_stage_metrics.csv`
- `alignment_stage_metrics_enriched.csv`
- `alignment_stage_summary_exact_bootstrap.csv`
- `all_metrics.csv`
- `all_metrics_enriched.csv`
- `final_stability_plasticity.csv`
- `final_group_summaries_exact_bootstrap.csv`
- `exact_paired_randomization_tests.csv`
- `direction_prediction_loso_by_seed.csv`
- `direction_vs_magnitude_predictive_tests.csv`
- `judge_validity.json`
- `behavior_judge_raw.csv`
- `behavior_judge_aggregate.csv`
- `behavior_judge_exact_paired_tests.csv`
- `final_stability_plasticity_with_judge.csv`
- `figures/*.pdf`

## 11. What not to do

- Do not install or replace the NVIDIA driver.
- Do not install a new PyTorch wheel unless the supplied PyTorch is itself broken.
- Do not run Fabric Manager on a single H100 PCIe instance.
- Do not embed Hugging Face or GitHub tokens in code.
- Do not delete phase checkpoints until results are backed up.

## 12. Protocol immutability

After a run has started, do not edit `config.json`, swap model revisions, or reuse the same output root for a modified experiment. The runner stores `STUDY_MANIFEST.json` and fails if the protocol, assets, or partitions no longer match. Use a new `--root` for a scientifically different run.
