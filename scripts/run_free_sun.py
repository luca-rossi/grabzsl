import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4115, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_free.py \
--dataset SUN --seed {args.seed} --features_per_class 300 --n_epochs 40 \
--n_critic_iters 1 --freeze_dec --lr 0.0002 --lr_cls 0.0005 --batch_size 512 \
--weight_critic 1 --weight_generator 1 --weight_precls 0.01 --weight_recons 0.1 \
--center_margin 120 --weight_margin 0.5 --weight_center 0.8'
if args.split:
	script += f' --split {args.split}'
os.system(script)
