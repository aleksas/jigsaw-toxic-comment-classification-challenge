PROBLEM=jigsaw_toxic_comment_classification
HPARAMS=transformer_multi_8000
MODEL=transformer_encoder
TRAIN_STEPS=30000
WORKER_GPU=1

USR_DIR=./jigsaw
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
PROBLEM_DIR=$HOME/t2t_train/$PROBLEM
TRAIN_DIR=$PROBLEM_DIR/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR

tensorboard --logdir $PROBLEM_DIR &

t2t-trainer \
 --generate_data \
 --problem=$PROBLEM \
 --train_steps=$TRAIN_STEPS \
 --data_dir=$DATA_DIR \
 --tmp_dir=$TMP_DIR \
 --output_dir=$TRAIN_DIR \
 --t2t_usr_dir=$USR_DIR \
 --hparams_set=$HPARAMS \
 --model=$MODEL \
 --worker_gpu=$WORKER_GPU
