import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=806, help='manual seed (for reproducibility)')
parser.add_argument('--split', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
args = parser.parse_args()
script = f'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_tfvaegan.py \
--dataset FLO --seed {args.seed} --features_per_class 1200 --n_epochs 80 \
--lr 0.0001 --lr_feedback 0.00001 --lr_decoder 0.0001 --lr_cls 0.001 \
--weight_gp 10 --weight_critic 10 --weight_generator 10 \
--weight_feed_train 0.5 --weight_feed_eval 0.5 --weight_recons 0.01'
if args.split:
	script += f' --split {args.split}'
os.system(script)
