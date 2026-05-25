Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose run --rm mamba2-vision python train.py `
  --dataset cifar10 `
  --data-dir /workspace/data `
  --output-dir /workspace/outputs/cifar10 `
  --image-size 32 `
  --patch-size 4 `
  --d-model 96 `
  --depth 2 `
  --d-state 32 `
  --headdim 32 `
  --epochs 5 `
  --batch-size 64 `
  --amp

