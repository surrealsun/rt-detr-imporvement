python train.py --config configs/experiments/baseline_aitod.yaml
python train.py --config configs/experiments/detail_aitod.yaml
python train.py --config configs/experiments/fusion_aitod.yaml
python train.py --config configs/experiments/aux_aitod.yaml
python train.py --config configs/experiments/full_aitod.yaml
python ablate.py --run-dirs outputs/aitod_baseline outputs/aitod_detail outputs/aitod_detail_fusion outputs/aitod_detail_aux outputs/aitod_full
