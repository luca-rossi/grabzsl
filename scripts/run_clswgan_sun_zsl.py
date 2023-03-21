import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4115, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset SUN --seed {args.seed} \
--n_attributes 102 --latent_size 102 --features_per_class 100 \
--n_epochs 54 --n_classes 40 --lr 0.0002 --lr_cls 0.0005 --weight_gp 10 --weight_precls 0.01'
if args.split:
	script += f' --split {args.split}'
os.system(script)
