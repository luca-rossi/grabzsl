import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset CUB --split '' --seed 3483 \
--n_attributes 312 --latent_size 312 --features_per_class 300 \
--n_epochs 56 --n_classes 200 --lr 0.0001 --lr_cls 0.001 --weight_gp 10 --weight_precls 0.01''')
