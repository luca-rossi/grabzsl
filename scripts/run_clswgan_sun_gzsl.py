import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_clswgan.py \
--dataset SUN --split '' --seed 4115 \
--n_attributes 102 --latent_size 102 --features_per_class 400 \
--n_epochs 40 --n_classes 717 --lr 0.0002 --lr_cls 0.001 --weight_gp 10 --weight_precls 0.01''')
