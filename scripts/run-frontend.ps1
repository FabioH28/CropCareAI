$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "frontend"

Set-Location $frontendRoot
npm.cmd run dev -- --host 127.0.0.1 --port 8080
