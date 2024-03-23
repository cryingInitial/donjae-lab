DATASET=cifar10
OPTIMIZER=sgd

METHOD=$1
SEEDS=(0 1 2)
# DEVICES=(0 1 2 3 4 5 6)
DEVICES=(2 3 4)
SEP='_'
SEED_SEP='seed'
TEST_MODE=class
MODEL_NAME=ResNet18
UNLEARN_EPOCHS=1
# LRS=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
LRS=(1e-5)
BATCH_SIZE=64
# class mode
CLASS_IDX=4
CLASS_UNLEARN=5000

# sample mode
SAMPLE_UNLEARN_PER_CLASS=100

mkdir -p watch
if [ $TEST_MODE = "sample" ]; then
    EXP_NAME=$METHOD$SEP$DATASET$SEP$TEST_MODE$SEP$SAMPLE_UNLEARN_PER_CLASS$SEP$UNLEARN_EPOCHS
elif [ $TEST_MODE = "class" ]; then
    EXP_NAME=$METHOD$SEP$DATASET$SEP$TEST_MODE$SEP$CLASS_IDX$SEP$CLASS_UNLEARN$SEP$UNLEARN_EPOCHS
else
    echo "TEST_MODE is not supported"
    exit
fi


for i in ${!DEVICES[@]}; do
    DEVICE=${DEVICES[$i]}
    LR=${LRS[$i]}
    NOTE=$OPTIMIZER'_'$LR
    for j in ${!SEEDS[@]}; do
        SEED=${SEEDS[$j]}
        CUDA_VISIBLE_DEVICES=$DEVICE python main.py --batch_size $BATCH_SIZE --rnd_seed $SEED --method $METHOD --data_name $DATASET --class_idx $CLASS_IDX --test_mode $TEST_MODE --model_name $MODEL_NAME --note $NOTE --lr $LR --optimizer $OPTIMIZER \
        --class_unlearn $CLASS_UNLEARN --sample_unlearn_per_class $SAMPLE_UNLEARN_PER_CLASS --unlearn_epochs $UNLEARN_EPOCHS --unlearn_aug --save_result_model > watch/$EXP_NAME$SEP$NOTE$SEP$SEED_SEP$SEED$SECOND_NOTE.log --unlearn_aug 2>&1 &
    done
done