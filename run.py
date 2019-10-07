import sys, os
from tensor2tensor.bin import t2t_trainer

if __name__ == '__main__':
    PROBLEM='jigsaw_toxic_comment_classification'
    HPARAMS='transformer_base_single'
    MODEL='transformer'

    USR_DIR='.'
    APPDATA=os.getenv('APPDATA')
    DATA_DIR='{}\\jigsaw\\t2t_data'.format(APPDATA)
    TMP_DIR='{}\\jigsaw\\tmp\\t2t_datagen'.format(APPDATA)
    TRAIN_DIR='{}\\jigsaw\\t2t_train\\{}\\{}-{}'.format(APPDATA, PROBLEM, MODEL, HPARAMS)
    
    try:
        os.mkdir( DATA_DIR )
    except FileExistsError:
        pass
    
    try:
        os.mkdir( TMP_DIR )
    except FileExistsError:
        pass

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