import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=3483, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset CUB --seed {args.seed} --features_per_class 300 --n_epochs 56 \
--lr 0.0001 --lr_cls 0.001 --weight_gp 10 --weight_precls 0.01'
if args.split:
	script += f' --split {args.split}'
os.system(script)
