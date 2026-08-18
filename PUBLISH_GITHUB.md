# Publishing the code and results to GitHub

Hugging Face authentication and GitHub authentication are separate. The Hugging Face read token is used only to fetch models and datasets. Authenticate to GitHub later with GitHub CLI or an SSH key.

## Recommended repository contents

Publish:

- all source files and shell launchers;
- `config.json`;
- `asset_manifest.json` with exact Hub revisions;
- `environment.lock.txt`;
- methods, reviewer map, and claim guardrails;
- aggregate CSV tables;
- final PDF figures;
- `STUDY_MANIFEST.json`, invocation manifests, and partition hashes.

Do not publish:

- `.venv/`;
- Hugging Face or GitHub credentials;
- Hub caches;
- phase-level LoRA checkpoints unless intentionally released elsewhere;
- raw logs that could contain credentials;
- downloaded base-model or judge-model weights.

The supplied `.gitignore` excludes the environment, caches, output checkpoints, tokens, and ordinary logs. Before the first push, inspect staged files:

```bash
git status
git diff --cached --stat
git diff --cached
```

A typical publication sequence after results are complete is:

```bash
gh auth login --hostname github.com --git-protocol https --web
git init
git add .
git status
git commit -m "Release AAAI-27 directional alignment experiment"
gh repo create aaai27-directional-alignment --public --source=. --push
```

Copy only the result tables and figures you intend to release from `pci_h100_outputs/results/`; the whole output directory is ignored by default because it also contains large checkpoints.
