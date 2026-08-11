[CmdletBinding()]
param(
    [ValidateSet("T2", "SELF")]
    [string]$Mode = "T2",

    [string[]]$Datasets = @(
        "adult", "bean", "default", "gesture", "letter", "magic", "news",
        "shoppers", "abalone", "bank", "breast", "california", "german", "mushroom"
    ),

    [string[]]$Splits = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),

    [string]$PythonExe = "C:\Users\DS\anaconda3\envs\daycon-env\python.exe",

    [string]$DataRoot = "C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar",

    [string]$OutputRoot = "",

    [string]$RunTag = "",

    [switch]$AllowFreshIncomplete,

    [switch]$ResumeOnly,

    [switch]$NoAmp,

    [ValidateRange(0, 64)]
    [int]$MaxConcurrentGpuJobs = 0,

    [string]$GpuSlotDirectory = "",

    [ValidateRange(1, 300)]
    [int]$GpuSlotPollSeconds = 5,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Runner = Join-Path $RepoRoot "run_eunseo_benchmark.py"

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Runner not found: $Runner"
}
if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if ($MaxConcurrentGpuJobs -gt 0) {
    if ([string]::IsNullOrWhiteSpace($GpuSlotDirectory)) {
        $GpuSlotDirectory = Join-Path $RepoRoot "results\.gpu_slots"
    }
    [System.IO.Directory]::CreateDirectory($GpuSlotDirectory) | Out-Null
}

$Datasets = @(
    foreach ($datasetValue in $Datasets) {
        foreach ($datasetName in ([string]$datasetValue -split ",")) {
            $trimmed = $datasetName.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                $trimmed
            }
        }
    }
)

$Splits = @(
    foreach ($splitValue in $Splits) {
        foreach ($splitText in ([string]$splitValue -split ",")) {
            $trimmed = $splitText.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                [int]$trimmed
            }
        }
    }
)

if ($Mode -eq "T2") {
    $MaskType = "MNAR_logistic_T2_v2_strict_hard"
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputRoot = Join-Path $RepoRoot "results\hard\t2"
    }
    if ([string]::IsNullOrWhiteSpace($RunTag)) {
        $RunTag = "eunseo_cat_wo_mpgr_hard_t2_e600_es"
    }
} else {
    $MaskType = "MNAR_self_logistic_v2_strict_hard"
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputRoot = Join-Path $RepoRoot "results\hard\self"
    }
    if ([string]::IsNullOrWhiteSpace($RunTag)) {
        $RunTag = "eunseo_cat_wo_mpgr_hard_self_e600_es"
    }
}

function Get-TaskBaseDir([string]$Dataset, [int]$Split) {
    return Join-Path (Join-Path (Join-Path (Join-Path (Join-Path $OutputRoot $Dataset) "rate30") $MaskType) ("split_{0}" -f $Split)) $RunTag
}

function Get-VersionDirs([string]$BaseDir) {
    if (-not (Test-Path -LiteralPath $BaseDir)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $BaseDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
}

function Test-CompleteReport([string]$ReportPath) {
    if (-not (Test-Path -LiteralPath $ReportPath)) {
        return $false
    }
    try {
        $reportText = [System.IO.File]::ReadAllText($ReportPath)
        return ($reportText -match '"test_metrics"\s*:')
    } catch {
        return $false
    }
}

function Test-TaskComplete([string]$Dataset, [int]$Split) {
    $baseDir = Get-TaskBaseDir $Dataset $Split
    foreach ($versionDir in Get-VersionDirs $baseDir) {
        if (Test-Path -LiteralPath (Join-Path $versionDir.FullName "test_pred_raw.npy")) {
            return $true
        }
        if (Test-CompleteReport (Join-Path $versionDir.FullName "report.json")) {
            return $true
        }
    }
    return $false
}

function Test-TaskHasAnyOutput([string]$Dataset, [int]$Split) {
    return (@(Get-VersionDirs (Get-TaskBaseDir $Dataset $Split)).Count -gt 0)
}

function Test-ReadableCheckpoint([string]$CheckpointPath) {
    if (-not (Test-Path -LiteralPath $CheckpointPath)) {
        return $false
    }
    $checkCode = "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $PythonExe -c $checkCode $CheckpointPath *> $null
        $checkpointExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return ($checkpointExitCode -eq 0)
}

function Find-ResumeCheckpoint([string]$Dataset, [int]$Split) {
    $baseDir = Get-TaskBaseDir $Dataset $Split
    foreach ($versionDir in Get-VersionDirs $baseDir) {
        foreach ($name in @("last_step.pt", "last.pt", "best.pt")) {
            $path = Join-Path (Join-Path $versionDir.FullName "checkpoints") $name
            if (Test-ReadableCheckpoint $path) {
                return $path
            }
            if (Test-Path -LiteralPath $path) {
                Write-Warning ("[skip-unreadable-checkpoint] {0}" -f $path)
            }
        }
    }
    return $null
}

function Enter-GpuSlot([string]$Dataset, [int]$Split) {
    if ($MaxConcurrentGpuJobs -le 0) {
        return $null
    }

    while ($true) {
        foreach ($slotIndex in 0..($MaxConcurrentGpuJobs - 1)) {
            $slotPath = Join-Path $GpuSlotDirectory ("gpu_slot_{0}.lock" -f $slotIndex)
            try {
                $stream = [System.IO.File]::Open(
                    $slotPath,
                    [System.IO.FileMode]::OpenOrCreate,
                    [System.IO.FileAccess]::ReadWrite,
                    [System.IO.FileShare]::None
                )
                $payload = "pid=$PID mode=$Mode split=$Split dataset=$Dataset acquired=$([DateTime]::Now.ToString('o'))"
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
                $stream.SetLength(0)
                $stream.Write($bytes, 0, $bytes.Length)
                $stream.Flush()

                Write-Host ("[gpu-slot-acquired] slot={0} split={1} dataset={2}" -f $slotIndex, $Split, $Dataset)
                return [PSCustomObject]@{
                    Index = $slotIndex
                    Path = $slotPath
                    Stream = $stream
                }
            } catch [System.IO.IOException] {
                continue
            }
        }

        Write-Host (
            "[gpu-slot-wait] split={0} dataset={1} active_limit={2}; no process is killed or restarted" -f
            $Split, $Dataset, $MaxConcurrentGpuJobs
        )
        Start-Sleep -Seconds $GpuSlotPollSeconds
    }
}

function Exit-GpuSlot($Slot) {
    if ($null -eq $Slot) {
        return
    }

    $slotIndex = $Slot.Index
    $Slot.Stream.Dispose()
    Write-Host ("[gpu-slot-released] slot={0}" -f $slotIndex)
}

Write-Host ("[run] Eunseo-Cat w/o MPGR hard {0} split-major resume-enabled" -f $Mode)
Write-Host ("[run] repo_root={0}" -f $RepoRoot)
Write-Host ("[run] datasets={0}" -f ($Datasets -join ", "))
Write-Host ("[run] splits={0}" -f ($Splits -join ", "))
Write-Host ("[run] mask={0}" -f $MaskType)
Write-Host ("[run] data_root={0}" -f $DataRoot)
Write-Host ("[run] output_root={0}" -f $OutputRoot)
Write-Host "[run] mpgr=false remasking=uniform_without_replacement ratio=0.25"
Write-Host "[run] max_epochs=600 patience=45 selection=valid-only"
if ($MaxConcurrentGpuJobs -gt 0) {
    Write-Host ("[run] gpu_concurrency_limit={0} slot_directory={1}" -f $MaxConcurrentGpuJobs, $GpuSlotDirectory)
} else {
    Write-Host "[run] gpu_concurrency_limit=disabled"
}

foreach ($split in $Splits) {
    Write-Host ""
    Write-Host ("=== Split {0} ===" -f $split)

    foreach ($dataset in $Datasets) {
        Write-Host ("[task] mode={0} split={1} dataset={2}" -f $Mode, $split, $dataset)

        if (Test-TaskComplete $dataset $split) {
            Write-Host ("[skip-completed] split={0} dataset={1}" -f $split, $dataset)
            continue
        }

        $checkpoint = Find-ResumeCheckpoint $dataset $split
        $pyArgs = @(
            $Runner,
            "--datasets", $dataset,
            "--splits", [string]$split,
            "--mask-type", $MaskType,
            "--ratio", "30",
            "--data-root", $DataRoot,
            "--output-root", $OutputRoot,
            "--run-tag", $RunTag,
            "--skip-completed",
            "--device", "cuda",
            "--max-epochs", "600",
            "--early-stop-patience", "45",
            "--early-stop-min-delta", "0.0002",
            "--selection-split", "valid"
        )

        if (-not $NoAmp) {
            $pyArgs += "--amp"
        }

        if ($checkpoint) {
            Write-Host ("[resume] split={0} dataset={1} checkpoint={2}" -f $split, $dataset, $checkpoint)
            $pyArgs += @("--resume-checkpoint", $checkpoint)
        } elseif ($ResumeOnly) {
            Write-Host ("[resume-only-skip-no-checkpoint] split={0} dataset={1}" -f $split, $dataset)
            continue
        } elseif (Test-TaskHasAnyOutput $dataset $split -and -not $AllowFreshIncomplete) {
            Write-Host ("[blocked-no-readable-checkpoint] split={0} dataset={1}; pass -AllowFreshIncomplete only for an intentional restart" -f $split, $dataset)
            continue
        } else {
            Write-Host ("[fresh] split={0} dataset={1}" -f $split, $dataset)
        }

        if ($DryRun) {
            Write-Host ("[dry-run] {0} {1}" -f $PythonExe, ($pyArgs -join " "))
            continue
        }

        $gpuSlot = $null
        $taskExitCode = -1
        try {
            $gpuSlot = Enter-GpuSlot $dataset $split
            & $PythonExe @pyArgs
            $taskExitCode = $LASTEXITCODE
        } finally {
            Exit-GpuSlot $gpuSlot
        }

        if ($taskExitCode -ne 0) {
            throw "Eunseo-Cat w/o MPGR failed: mode=$Mode split=$split dataset=$dataset exit_code=$taskExitCode"
        }
    }
}

Write-Host "[done] Eunseo-Cat w/o MPGR split-major run finished."
