PROBLEM=jigsaw_toxic_comment_classification_characters
HPARAMS=transformer_base
MODEL=transformer

USR_DIR=./jigsaw
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR

tensorboard --logdir $TRAIN_DIR &

t2t-trainer \
 --generate_data \
 --problem=$PROBLEM \
 --data_dir=$DATA_DIR \
 --tmp_dir=$TMP_DIR \
 --output_dir=$TRAIN_DIR \
 --t2t_usr_dir=$USR_DIR \
 --hparams_set=$HPARAMS \
 --model=$MODEL
