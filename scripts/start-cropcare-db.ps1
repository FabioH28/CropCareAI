$repoRoot = Split-Path -Parent $PSScriptRoot
$myIni = Join-Path $repoRoot "database\\xampp-local\\my.ini"
$pidFile = Join-Path $repoRoot "database\\xampp-local\\mysql.pid"

if (Test-Path $pidFile) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
  $processExists = $false

  if ($existingPid) {
    try {
      $null = Get-Process -Id ([int]$existingPid) -ErrorAction Stop
      $processExists = $true
    } catch {
      $processExists = $false
    }
  }

  if (-not $processExists) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  }
}

Start-Process -FilePath "C:\\xampp\\mysql\\bin\\mysqld.exe" `
  -ArgumentList "--defaults-file=$myIni", "--standalone" `
  -WorkingDirectory "C:\\xampp"

Write-Host "Started the project-local CropCare MariaDB instance on port 3308."
