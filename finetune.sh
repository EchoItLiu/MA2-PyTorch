# Set the path to save checkpoints
OUTPUT_DIR='/home/liullhappy/imageNet/MAE-pytorch-main-ablation/output/FT'
DATA_PATH='/home/liullhappy/imageNet'
# path to pretrain model
# MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main/output/pretrain_mae_base_patch16_224/checkpoint-2.pth'
# MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main/output/pretrain_mae_base_patch16_224/checkpoint-399.pth'

MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main-ablation/output/PT/checkpoint-9.pth'

# path to tensorboard
LOG_DIR='/home/liullhappy/imageNet/MAE-pytorch-main/output/FT_results'

# path to confusion matrix
CM_DIR='/home/liullhappy/imageNet/MAE-pytorch-main/cfm'

# batch_size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 --master_port=8675 --use_env run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --cm_dir ${CM_DIR} \
    --batch_size 32 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 30 \
    --dist_eval

    