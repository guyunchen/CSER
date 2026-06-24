param(
    [string]$Message = "update cser code",
    [string]$RemoteUrl = "",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not (Test-Path ".git")) {
    git init
}

$currentBranch = git branch --show-current
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    git checkout -b $Branch
} elseif ($currentBranch -ne $Branch) {
    git branch -M $Branch
}

$hasRemote = $false
try {
    $origin = git remote get-url origin
    if (-not [string]::IsNullOrWhiteSpace($origin)) {
        $hasRemote = $true
    }
} catch {
    $hasRemote = $false
}

if (-not [string]::IsNullOrWhiteSpace($RemoteUrl)) {
    if ($hasRemote) {
        git remote set-url origin $RemoteUrl
    } else {
        git remote add origin $RemoteUrl
    }
    $hasRemote = $true
}

git status --short
git add -A

$staged = git diff --cached --name-only
if ([string]::IsNullOrWhiteSpace($staged)) {
    Write-Host "No staged changes to commit."
} else {
    git commit -m $Message
}

if ($hasRemote) {
    git push -u origin $Branch
} else {
    Write-Host "No origin remote configured. Re-run with -RemoteUrl https://github.com/<user>/<repo>.git"
}
