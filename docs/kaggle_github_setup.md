# Kaggle GitHub Workflow

Use this flow to avoid re-uploading a zip after every code change.

## Local Sync

First time:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_to_github.ps1 `
  -RemoteUrl https://github.com/<user>/<repo>.git `
  -Message "initial cser code"
```

After later code edits:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_to_github.ps1 `
  -Message "update robust frontend"
```

## Kaggle Notebook: Public Repo

```bash
rm -rf /kaggle/working/CSER
git clone https://github.com/<user>/<repo>.git /kaggle/working/CSER
cd /kaggle/working/CSER
git rev-parse --short HEAD
```

## Kaggle Notebook: Private Repo

Create a Kaggle Secret named `GITHUB_TOKEN`, then use:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo = "github.com/<user>/<repo>.git"
!rm -rf /kaggle/working/CSER
!git clone https://{token}@{repo} /kaggle/working/CSER
%cd /kaggle/working/CSER
!git rev-parse --short HEAD
```

## Run Robust Frontend Experiment

```bash
python train_loso.py --config experiments/configs/robust_frontend.yaml --only-ids R0 R1 R2 --seeds 42 --num-workers 2
python experiments/summarize_results.py
tar -czf /kaggle/working/cser_outputs.tar.gz experiments/output
```

For a quick smoke test:

```bash
python train_loso.py --config experiments/configs/robust_frontend.yaml --only-ids R1 --folds 0 --epochs 1 --max-batches 1 --num-workers 2
```
