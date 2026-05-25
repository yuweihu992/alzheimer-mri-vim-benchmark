param(
  [Parameter(Mandatory = $true)][string]$Checkpoint,
  [Parameter(Mandatory = $true)][string]$Image,
  [int]$TopK = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose run --rm mamba2-vision python infer.py `
  --checkpoint $Checkpoint `
  --image $Image `
  --top-k $TopK

