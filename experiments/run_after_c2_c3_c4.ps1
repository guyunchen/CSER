$ErrorActionPreference = "Stop"

$root = "E:\SER\CSER"
$python = "D:\Anaconda3\envs\CSER\python.exe"
$c2Pids = @(28504, 45040)
$c2Result = Join-Path $root "experiments\output\results\ablations_C2_disable_ls4_dynamic_seed42_20260605_204522.json"
$logPath = Join-Path $root "experiments\output\logs\run_after_c2_c3_c4.log"
$lockPath = Join-Path $root "experiments\output\logs\run_after_c2_c3_c4.lock"

Set-Location $root
if (Test-Path $lockPath) {
    "[$(Get-Date -Format s)] Lock exists; another handoff is already active. Exiting." | Out-File -FilePath $logPath -Encoding utf8 -Append
    exit 0
}

New-Item -Path $lockPath -ItemType File -Force | Out-Null
"[$(Get-Date -Format s)] Waiting for C2 to finish..." | Out-File -FilePath $logPath -Encoding utf8

try {
    while ($true) {
        if (Test-Path $c2Result) {
            "[$(Get-Date -Format s)] C2 result found: $c2Result" | Out-File -FilePath $logPath -Encoding utf8 -Append
            break
        }

        $running = @()
        foreach ($pidValue in $c2Pids) {
            $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($proc) {
                $running += $proc
            }
        }

        if ($running.Count -eq 0) {
            "[$(Get-Date -Format s)] C2 processes ended; continuing even though result file was not found yet." | Out-File -FilePath $logPath -Encoding utf8 -Append
            break
        }

        Start-Sleep -Seconds 60
    }

    "[$(Get-Date -Format s)] Starting C3 and C4..." | Out-File -FilePath $logPath -Encoding utf8 -Append
    & $python "experiments\run_experiments.py" --config "experiments\configs\ablations.yaml" --only-ids C3 C4 --seeds 42 --num-workers 0 --skip-followups 2>&1 |
        Tee-Object -FilePath $logPath -Append

    "[$(Get-Date -Format s)] C3 and C4 launcher finished." | Out-File -FilePath $logPath -Encoding utf8 -Append
}
finally {
    Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
}
