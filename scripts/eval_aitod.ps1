param(
    [string]$Checkpoint = "outputs/aitod_full/best.pth"
)

python eval.py --config configs/experiments/full_aitod.yaml --checkpoint $Checkpoint
