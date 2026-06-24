param(
    [string]$Output = "CSER_colab_package.zip",
    [switch]$IncludeOutputs,
    [switch]$SlimDataset,
    [switch]$CodeOnly
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputPath = Join-Path $root $Output
$tempDir = Join-Path $env:TEMP ("CSER_colab_package_" + [guid]::NewGuid().ToString("N"))
$stageDir = Join-Path $tempDir "CSER"

New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$excludeDirs = @(".idea", "__pycache__", ".git")
if (-not $IncludeOutputs) {
    $excludeDirs += @("output", "outputs", "training_logs")
}

$excludeFiles = @("*.pyc", "*.pyo", "*.zip")

Get-ChildItem -Path $root -Force | ForEach-Object {
    if ($excludeDirs -contains $_.Name) {
        return
    }
    $dest = Join-Path $stageDir $_.Name
    if ($CodeOnly -and $_.Name -eq "dataset") {
        return
    }
    if ($_.PSIsContainer -and $_.Name -eq "dataset" -and $SlimDataset) {
        $finalData = Join-Path $_.FullName "IEMOCAP\final_data"
        $destFinalData = Join-Path $dest "IEMOCAP\final_data"
        New-Item -ItemType Directory -Force -Path $destFinalData | Out-Null
        Copy-Item -LiteralPath (Join-Path $finalData "train_v3.parquet") -Destination $destFinalData -Force
        Copy-Item -LiteralPath (Join-Path $finalData "test_v3.parquet") -Destination $destFinalData -Force
    } elseif ($_.PSIsContainer) {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    } else {
        $skip = $false
        foreach ($pattern in $excludeFiles) {
            if ($_.Name -like $pattern) {
                $skip = $true
            }
        }
        if (-not $skip) {
            Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
        }
    }
}

if (Test-Path $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

Compress-Archive -Path $stageDir -DestinationPath $outputPath -Force
Remove-Item -LiteralPath $tempDir -Recurse -Force

Write-Host "Created package: $outputPath"
Write-Host "Upload this zip to Google Drive, then set PROJECT_ZIP in the Colab notebook."
if ($SlimDataset) {
    Write-Host "SlimDataset enabled: only dataset/IEMOCAP/final_data/train_v3.parquet and test_v3.parquet were included."
}
if ($CodeOnly) {
    Write-Host "CodeOnly enabled: dataset was excluded."
}
