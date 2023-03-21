import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=806, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset FLO --seed {args.seed} \
--n_attributes 1024 --latent_size 1024 --features_per_class 300 \
--n_epochs 97 --n_classes 102 --lr 0.0001 --weight_gp 10 --weight_precls 0.1'
if args.split:
	script += f' --split {args.split}'
os.system(script)
