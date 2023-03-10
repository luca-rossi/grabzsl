import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_tfvaegan.py \
--dataset CUB --split '' --seed 3483 \
--n_attributes 312 --latent_size 312 --features_per_class 300 \
--n_epochs 300 --n_classes 200 \
--lr 0.0001 --lr_feedback 0.00001 --lr_decoder 0.0001 --lr_cls 0.001 \
--weight_gp 10 --weight_critic 10 --weight_generator 10 \
--weight_feed_train 1 --weight_feed_eval 1 --weight_recons 0.1''')
