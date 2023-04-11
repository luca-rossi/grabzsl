import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=3483, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_free.py \
--dataset CUB --seed {args.seed} --features_per_class 700 --n_epochs 56 \
--n_critic_iters 1 --freeze_dec --lr 0.0001 \
--weight_critic 10 --weight_generator 10 --weight_precls 0.01 --weight_recons 0.001 \
--center_margin 200 --weight_margin 0.5 --weight_center 0.8'
if args.split:
	script += f' --split {args.split}'
os.system(script)
