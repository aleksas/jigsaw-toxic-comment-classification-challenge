import sys, os
from tensor2tensor.bin import t2t_trainer
import pathlib

if __name__ == '__main__':
    PROBLEM='jigsaw_toxic_comment_classification'
    HPARAMS='transformer_base_single'
    MODEL='transformer'

    USR_DIR='./jigsaw-toxic-comment-classification-challenge'
    DATA_DIR='/tmp/jigsaw/t2t_data'
    TMP_DIR='/tmp/jigsaw/t2t_datagen'
    TRAIN_DIR='/tmp/jigsaw/t2t_train/{}/{}-{}'.format(PROBLEM, MODEL, HPARAMS)
    
    pathlib.Path( DATA_DIR ).mkdir(parents=True, exist_ok=True)
    pathlib.Path( TMP_DIR ).mkdir(parents=True, exist_ok=True)

    argv = [
        '--generate_data',
        '--problem', PROBLEM,
        '--data_dir', DATA_DIR,
        '--tmp_dir', TMP_DIR,
        '--output_dir', TRAIN_DIR,
        '--t2t_usr_dir', USR_DIR,
        '--hparams_set', HPARAMS,
        '--model', MODEL
    ]
    sys.argv += argv

    t2t_trainer.main(None)