import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_tfvaegan.py \
--dataset AWA --split '' --seed 9182 \
--n_attributes 85 --latent_size 85 --features_per_class 1800 \
--n_epochs 120 --n_classes 50 --freeze_dec \
--lr 0.00001 --lr_feedback 0.0001 --lr_decoder 0.0001 --lr_cls 0.001 \
--weight_gp 10 --weight_critic 10 --weight_generator 10 \
--weight_feed_train 0.01 --weight_feed_eval 0.01 --weight_recons 0.1''')
