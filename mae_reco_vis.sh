# Set the path to save images
OUTPUT_DIR='/home/liullhappy/imageNet/MAE-pytorch-main/re_output'
# path to image for visualization
IMAGE_PATH='/home/liullhappy/imageNet/test/ILSVRC2012_test_00049999.JPEG'
# path to pretrain model
# MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main/output/checkpoint-best.pth'
# MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main/output/checkpoint-best.pth'
MODEL_PATH='/home/liullhappy/imageNet/MAE-pytorch-main/output/pretrain_mae_base_patch16_224/checkpoint-399.pth'
# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}