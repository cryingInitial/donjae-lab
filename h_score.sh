# DEVICES=(3 4 5)
# DEVICES=(2 6 7)
DEVICES=(5 6 7)
SEEDS=(0 1 2)
METHOD="lp"
DATASET="cifar10"
DL=5000

NAME=$METHOD"_"$DATASET
# idx the length of devices
mkdir -p hscore/$NAME
for i in ${!DEVICES[@]}; do
    DEVICE=${DEVICES[$i]}
    SEED=${SEEDS[$i]}
    MODEL_PATH="../../jwj/MUA3/checkpoints/"$METHOD"_"$DATASET"_class_4_"$DL"_1.0_"$SEED".pth"
    # MODEL_PATH="../../jwj/MUA3/checkpoints/ResNet18_cifar100_class_4_500_retrain.pth"

    # MODEL_PATH="checkpoints/scrub_cifar10_class_4_5000_1.0_adam_5e-5$SEED.pth"
    # MODEL_PATH="checkpoints/ft_cifar10_class_4_5000_1.0_0_50000.pth"
    # MODEL_PATH="cifar100_model/ResNet18_cifar100_sgd_seed"$SEED"_ori.pth"
    NOTE=$NAME
    CUDA_VISIBLE_DEVICES=$DEVICE python3 metric_plz_binary_last.py --extractor_path $MODEL_PATH --data_name $DATASET --dataset_length $DL > hscore/$NAME/$SEED.log 2>&1 &
done
