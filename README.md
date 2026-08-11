# Eunseo-Cat without MPGR

This repository is the isolated `w/o MPGR` ablation of Eunseo-Cat for
tabular-data imputation under MNAR missingness.

Only MPGR is removed. TOSE, the native categorical head, all losses, data
splits, metrics, seeds, validation-only checkpoint selection, and the training
schedule are held fixed relative to Eunseo-Cat.

## What Changed

Eunseo-Cat normally uses missing-pattern guided remasking (MPGR). For each row,
the co-missingness matrix scores the currently observed logical features and
preferentially chooses auxiliary reconstruction targets.

This ablation replaces that choice with uniform sampling without replacement:

```text
Eunseo-Cat: observed feature -> co-missingness score -> sample 25%
w/o MPGR:   observed feature -> uniform random sampling -> sample 25%
```

The remasking ratio remains `0.25`. Natural missing cells are never used as
training targets; only observed logical features selected by the auxiliary
mask provide self-supervised targets.

## What Is Unchanged

- Target-conditional observed-set encoder (TOSE)
- Target query and observed-memory cross-attention
- Numeric decoder and normalized numeric training space
- Per-column native-category CatHead
- Direct categorical cross-entropy and codebook/bit auxiliary losses
- Hard MNAR train/valid/test files
- `seed = 20260430 + split_idx`
- Maximum 600 epochs and early-stopping patience 45
- Validation-only best-checkpoint selection
- Selection score: `RMSE - 0.1 * weighted_F1 - 0.05 * accuracy`
- Numeric RMSE/MAE in the prepared `(x - mean) / std / 2.0` space
- Native category-index decoding and the common benchmark metric schema

## Repository Layout

```text
.
|-- config.py
|-- dataset.py
|-- losses.py
|-- metrics.py
|-- model.py
|-- run_eunseo_benchmark.py
|-- trainer.py
|-- scripts/
|   |-- run_hard_split_major.ps1
|   `-- launch_hard_10_processes.ps1
`-- results/                       # generated locally and ignored by Git
```

## Environment

```powershell
Set-Location "C:\Users\DS\eunseo\Eunseo-Github\ablation_studies\Eunseo-Cat-WoMPGR"

C:\Users\DS\anaconda3\envs\daycon-env\python.exe -m pip install -r .\requirements.txt
```

The benchmark data are shared with the main workspace:

```text
C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar
```

## Run One Split

The following command runs all 14 datasets sequentially for T2 hard split 0.
Completed tasks are skipped, and readable checkpoints are resumed.

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_hard_split_major.ps1" `
  -Mode T2 `
  -Splits 0 `
  -DataRoot "C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar" `
  -PythonExe "C:\Users\DS\anaconda3\envs\daycon-env\python.exe"
```

## Run Ten Visible Processes

This launches ten visible PowerShell windows. Each window owns exactly one T2
split and processes all 14 datasets sequentially. To avoid exhausting an 8 GB
GPU and host RAM, at most eight training jobs enter the GPU section at once;
the remaining visible windows wait for a file-lock slot without changing the
model, batch size, loss, or checkpoint protocol.

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\launch_hard_10_processes.ps1" `
  -Mode T2 `
  -MaxConcurrentGpuJobs 8 `
  -DataRoot "C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar" `
  -PythonExe "C:\Users\DS\anaconda3\envs\daycon-env\python.exe"
```

No watchdog or inactivity timeout is used. A process is not killed and
restarted merely because an epoch takes a long time. GPU slots use OS file
locks, so an unexpectedly closed process releases its slot automatically.

If a task failed before writing any readable checkpoint, intentionally start
a new version with:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_hard_split_major.ps1" `
  -Mode T2 `
  -Splits 9 `
  -AllowFreshIncomplete `
  -DataRoot "C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar" `
  -PythonExe "C:\Users\DS\anaconda3\envs\daycon-env\python.exe"
```

When a readable checkpoint exists, the same runner resumes it instead of
starting a new version.

## Results

T2 results are written under:

```text
results/hard/t2/<dataset>/rate30/MNAR_logistic_T2_v2_strict_hard/
  split_<k>/eunseo_cat_wo_mpgr_hard_t2_e600_es/v*/
```

Generated data, checkpoints, predictions, plots, and result files are excluded
from Git.
