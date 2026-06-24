$ErrorActionPreference = "SilentlyContinue"

$root = "E:\SER\CSER"
$parentPid = 39664
$currentResult = Join-Path $root "experiments\output\results\core_models_A4_original_ls4_seed3407_20260607_065031.json"
$nextResult = Join-Path $root "experiments\output\results\core_models_A4_original_ls4_seed2026_20260607_065031.json"
$monitorLog = Join-Path $root "experiments\output\logs\monitor_stop_a4_20260607_065031.log"

function Write-MonitorLog($message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $monitorLog -Value "$stamp $message"
}

Write-MonitorLog "Monitoring seed3407 result. Parent PID: $parentPid"

while ($true) {
    if (Test-Path $currentResult) {
        try {
            $result = Get-Content $currentResult -Raw | ConvertFrom-Json
            $meanCcc = [double]$result.mean_ccc
            Write-MonitorLog "seed3407 finished with mean_ccc=$meanCcc"
            if ($meanCcc -le 0.63) {
                Write-MonitorLog "mean_ccc <= 0.63; stopping remaining A4 run."
                Stop-Process -Id $parentPid -Force
                $skipped = [ordered]@{
                    experiment = "original_ls4"
                    group = "core_models"
                    id = "A4"
                    seed = 2026
                    status = "skipped"
                    reason = "Stopped after seed3407 because best mean_ccc <= 0.63."
                    previous_seed = 3407
                    previous_seed_mean_ccc = $meanCcc
                    config_note = "A4 optimizer lr has been updated to 0.001 for future runs."
                    finished_at = (Get-Date).ToString("s")
                }
                $skipped | ConvertTo-Json -Depth 4 | Set-Content -Path $nextResult -Encoding UTF8
            } else {
                Write-MonitorLog "mean_ccc > 0.63; allowing remaining seeds to continue."
            }
        } catch {
            Write-MonitorLog "Failed to process result: $($_.Exception.Message)"
        }
        break
    }
    Start-Sleep -Seconds 10
}

Write-MonitorLog "Monitor exiting."
