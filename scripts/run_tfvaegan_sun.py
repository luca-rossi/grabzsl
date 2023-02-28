import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_tfvaegan.py \
--dataset SUN --split '' --seed 4115 \
--n_attributes 102 --latent_size 102 --features_per_class 400 \
--n_epochs 400 --n_classes 717 \
--lr 0.001 --lr_feedback 0.0001 --lr_cls 0.0005 \
--weight_gp 10 --weight_critic 1 --weight_generator 1 \
--weight_feed_train 0.1 --weight_feed_eval 0.01 --weight_recons 0.01''')
