$repoRoot = Split-Path -Parent $PSScriptRoot
$backendRoot = Join-Path $repoRoot "backend"

Set-Location $backendRoot
& ".\\.venv\\Scripts\\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
