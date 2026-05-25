Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose build

