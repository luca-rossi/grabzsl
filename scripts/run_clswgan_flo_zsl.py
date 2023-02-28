import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset FLO --split '' --seed 806 \
--n_attributes 1024 --latent_size 1024 --features_per_class 300 \
--n_epochs 97 --n_classes 102 --lr 0.0001 --weight_gp 10 --weight_precls 0.1''')
