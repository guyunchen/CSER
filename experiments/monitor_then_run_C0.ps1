param(
    [int[]]$WatchPids = @(53016, 65348),
    [string]$WatchResult = "E:\SER\CSER\experiments\output\results\core_models_A3_ls4_gated_robust_seed42_20260604_174922.json",
    [string]$Root = "E:\SER\CSER",
    [string]$Python = "D:\Anaconda3\envs\CSER\python.exe"
)

$ErrorActionPreference = "Stop"
$logDir = Join-Path $Root "experiments\output\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$monitorLog = Join-Path $logDir "monitor_then_run_C0.log"
$lockPath = Join-Path $logDir "monitor_then_run_C0.lock"

function Write-MonitorLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $monitorLog -Value "$stamp $Message"
}

if (Test-Path $lockPath) {
    $existing = Get-Content -LiteralPath $lockPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing -and (Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue)) {
        Write-MonitorLog "Another monitor is already active with PID $existing. Exiting."
        exit 0
    }
}

Set-Content -LiteralPath $lockPath -Value $PID
Write-MonitorLog "Monitor started. Watching PIDs: $($WatchPids -join ', '); result: $WatchResult"

try {
    while ($true) {
        $resultDone = Test-Path -LiteralPath $WatchResult
        $alive = @()
        foreach ($watchPid in $WatchPids) {
            if (Get-Process -Id $watchPid -ErrorAction SilentlyContinue) {
                $alive += $watchPid
            }
        }

        if ($resultDone -or $alive.Count -eq 0) {
            if ($resultDone) {
                Write-MonitorLog "Watched result file exists. Starting C0."
            } else {
                Write-MonitorLog "Watched PIDs have exited. Starting C0."
            }
            break
        }

        Write-MonitorLog "Still waiting. Alive PIDs: $($alive -join ', ')"
        Start-Sleep -Seconds 60
    }

    Set-Location $Root
    $args = @(
        "experiments\run_experiments.py",
        "--config", "experiments\configs\ablations.yaml",
        "--only-ids", "C0",
        "--num-workers", "0",
        "--skip-followups"
    )
    Write-MonitorLog "Running: $Python $($args -join ' ')"
    & $Python @args 2>&1 | Tee-Object -FilePath (Join-Path $logDir "monitor_started_C0.stdout.log")
    $exitCode = $LASTEXITCODE
    Write-MonitorLog "C0 finished with exit code $exitCode."
    exit $exitCode
}
finally {
    if (Test-Path $lockPath) {
        Remove-Item -LiteralPath $lockPath -Force
    }
}
