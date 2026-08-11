[CmdletBinding()]
param(
    [ValidateSet("T2", "SELF")]
    [string]$Mode = "T2",

    [string[]]$Datasets = @(
        "adult", "bean", "default", "gesture", "letter", "magic", "news",
        "shoppers", "abalone", "bank", "breast", "california", "german", "mushroom"
    ),

    [string]$PythonExe = "C:\Users\DS\anaconda3\envs\daycon-env\python.exe",

    [string]$DataRoot = "C:\Users\DS\eunseo\Eunseo-Github\data\hard_mnar",

    [ValidateRange(1, 10)]
    [int]$MaxConcurrentGpuJobs = 8,

    [int]$StartDelaySeconds = 2,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Runner = Join-Path $RepoRoot "scripts\run_hard_split_major.ps1"
$GpuSlotDirectory = Join-Path $RepoRoot "results\.gpu_slots"

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Runner not found: $Runner"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}

$datasetArgument = $Datasets -join ","
$launched = @()

Write-Host "[launch] Eunseo-Cat w/o MPGR, Direct-CE-only"
Write-Host ("[launch] mode={0} processes=10 split_ownership=one-split-per-window" -f $Mode)
Write-Host ("[launch] repo_root={0}" -f $RepoRoot)
Write-Host ("[launch] data_root={0}" -f $DataRoot)
Write-Host "[launch] watchdog=false inactivity_timeout=false"
Write-Host ("[launch] visible_windows=10 max_concurrent_gpu_jobs={0}" -f $MaxConcurrentGpuJobs)

foreach ($split in 0..9) {
    $title = "Eunseo-Cat w/o MPGR DirectCE $Mode split $split"
    $command = (
        "& '{0}' -Mode '{1}' -Splits '{2}' -Datasets '{3}' -PythonExe '{4}' -DataRoot '{5}' -MaxConcurrentGpuJobs '{6}' -GpuSlotDirectory '{7}'" -f
        $Runner, $Mode, $split, $datasetArgument, $PythonExe, $DataRoot, $MaxConcurrentGpuJobs, $GpuSlotDirectory
    )

    if ($DryRun) {
        Write-Host ("[dry-run] split={0} command={1}" -f $split, $command)
        continue
    }

    $process = Start-Process -FilePath "powershell.exe" `
        -WorkingDirectory $RepoRoot `
        -ArgumentList @(
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command", "[Console]::Title='$title'; $command"
        ) `
        -PassThru

    $launched += [PSCustomObject]@{
        Split = $split
        ProcessId = $process.Id
        Title = $title
    }
    Write-Host ("[launched] split={0} pid={1} title={2}" -f $split, $process.Id, $title)
    Start-Sleep -Seconds $StartDelaySeconds
}

if (-not $DryRun) {
    $launched | Format-Table -AutoSize
    Write-Host ("[done] launched={0}; each training process is foreground in its own visible PowerShell window" -f $launched.Count)
}
