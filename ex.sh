SEEDS=(0 1 2)
# DEVICES=(0 1 2)
DEVICES=(3 4 5)
METHOD=$1
DATASET=cifar10
SEP='_'
SEED_SEP='seed'
TEST_MODE=class
MODEL_NAME=ResNet18
UNLEARN_EPOCHS=1

# class mode
CLASS_IDX=4
CLASS_UNLEARN=5000

# sample mode
SAMPLE_UNLEARN_PER_CLASS=100

# assert that the length of SEEDS and DEVICES are the same
if [ ${#SEEDS[@]} -ne ${#DEVICES[@]} ]; then
    echo "Length of SEEDS and DEVICES must be the same!"
    exit 1
fi

mkdir -p watch
# if sample mode
if [ $TEST_MODE = "sample" ]; then
    EXP_NAME=$METHOD$SEP$DATASET$SEP$TEST_MODE$SEP$SAMPLE_UNLEARN_PER_CLASS$SEP$UNLEARN_EPOCHS
elif [ $TEST_MODE = "class" ]; then
    EXP_NAME=$METHOD$SEP$DATASET$SEP$TEST_MODE$SEP$CLASS_IDX$SEP$CLASS_UNLEARN$SEP$UNLEARN_EPOCHS
else
    echo "TEST_MODE is not supported"
    exit
fi


for i in ${!SEEDS[@]}; do
    SEED=${SEEDS[$i]}
    DEVICE=${DEVICES[$i]}
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py --rnd_seed $SEED --method $METHOD --data_name $DATASET --class_idx $CLASS_IDX --test_mode $TEST_MODE --model_name $MODEL_NAME\
        --class_unlearn $CLASS_UNLEARN --sample_unlearn_per_class $SAMPLE_UNLEARN_PER_CLASS --unlearn_epochs $UNLEARN_EPOCHS > watch/$EXP_NAME$SEP$SEED_SEP$SEED.log 2>&1 &
done

# CUDA_VISIBLE_DEVICES=$DEVICE python main.py --method $METHOD --data_name $DATASET --class_idx $CLASS_IDX --test_mode $TEST_MODE --model_name $MODEL_NAME\
#     --class_unlearn $CLASS_UNLEARN --sample_unlearn_per_class $SAMPLE_UNLEARN_PER_CLASS --unlearn_epochs $UNLEARN_EPOCHS > watch/$EXP_NAME.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --rnd_seed 0 --method sparse --data_name cifar10 --class_idx 4 --test_mode class --model_name ResNet18  --class_unlearn 5000 --sample_unlearn_per_class 100 --unlearn_epochs 1
