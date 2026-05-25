param(
  [string]$TrainDir = "/workspace/data/imagefolder/train",
  [string]$ValDir = "/workspace/data/imagefolder/val",
  [string]$ModelArch = "vim",
  [int]$ImageSize = 384,
  [int]$Epochs = 20,
  [int]$BatchSize = 32,
  [int]$Workers = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose run --rm mamba2-vision python train.py `
  --dataset imagefolder `
  --train-dir $TrainDir `
  --val-dir $ValDir `
  --output-dir /workspace/outputs/imagefolder `
  --model-arch $ModelArch `
  --image-size $ImageSize `
  --patch-size 16 `
  --d-model 192 `
  --depth 6 `
  --d-state 64 `
  --headdim 64 `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --workers $Workers `
  --amp
