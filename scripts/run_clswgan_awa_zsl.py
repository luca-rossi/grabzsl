import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--split', '-s', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
os.system(f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset AWA2 --split \'{args.split}\' --seed 9182 \
--n_attributes 85 --latent_size 85 --features_per_class 300 \
--n_epochs 30 --n_classes 50 --lr 0.00001 --weight_gp 10 --weight_precls 0.01')
