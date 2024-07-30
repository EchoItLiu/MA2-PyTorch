# Set the path to save checkpoints
OUTPUT_DIR='/home/liullhappy/imageNet/MAE-pytorch-main-ablation/output/PT'
# path to imagenet-1k train set
DATA_PATH='/home/liullhappy/imageNet/train'
# Set the path to save TesnorBoard --log_dir
LOG_DIR='/home/liullhappy/imageNet/MAE-pytorch-main/log_output'

# batch_size can be adjusted according to the graphics card
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 run_mae_pretraining.py \
#         --data_path ${DATA_PATH} \cc
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch16_224 \
#         --batch_size 64 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --epochs 16 \
#         --output_dir ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=8675 --use_env run_mae_pretraining.py \
#         --data_path ${DATA_PATH} \
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch16_224 \
#         --batch_size 64 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --epochs 16 \
#         --output_dir ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 100 \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${LOG_DIR}


    