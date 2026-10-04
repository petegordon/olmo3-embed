#!/usr/bin/env bash
# Kill any python.exe whose working set exceeds LIMIT_GB (default 8). Guards against Windows CUDA sysmem fallback
# silently eating system RAM. Usage: scripts/ram_guard.sh [LIMIT_GB] &   (stop with kill)
LIMIT=${1:-8}
while true; do
  powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { \$_.WorkingSet64/1GB -gt $LIMIT } | ForEach-Object { Write-Output (\"RAM_GUARD killed pid {0} at {1:N1} GB\" -f \$_.Id, (\$_.WorkingSet64/1GB)); Stop-Process -Id \$_.Id -Force }"
  sleep 10
done
