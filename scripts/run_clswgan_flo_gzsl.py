import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=806, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset FLO --seed {args.seed} --features_per_class 1200 --n_epochs 80 \
--lr 0.0001 --lr_cls 0.001 --weight_gp 10 --weight_precls 0.1'
if args.split:
	script += f' --split {args.split}'
os.system(script)
