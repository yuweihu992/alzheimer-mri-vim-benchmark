Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose run --rm mamba2-vision python tools/smoke_test.py

