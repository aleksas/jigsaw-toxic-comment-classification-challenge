import sys, os
from tensor2tensor.bin import t2t_trainer

if __name__ == '__main__':
    PROBLEM='jigsaw_toxic_comment_classification'
    HPARAMS='transformer_base_single'
    MODEL='transformer'

    USR_DIR='.'
    DATA_DIR='%APPDATA%/jigsaw/t2t_data'
    TMP_DIR='%APPDATA%/jigsaw/tmp/t2t_datagen'
    TRAIN_DIR='%APPDATA%/jigsaw/t2t_train/%s/%s-%s' % (PROBLEM, MODEL, HPARAMS)
    
    os.mkdir( DATA_DIR )
    os.mkdir( TMP_DIR )

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