import os

ROOT = os.curdir
PHASE = ['train', 'test']
NUM_CLASSES = 62
DATA_INFO = {
    'train': {
        'image_path': [],
        'class_id': [],
    },
    'test': {
        'image_path': [],
        'class_id': [],
    }
}

ANNOTATION_FILE_NAME = '%s_annotation.csv'

TRANSFORM_RESIZE_X = 500
TRANSFORM_RESIZE_Y = 500

BATCH_SIZE = 10

LEARNING_RATE = 0.01

MOMENTUM = 0.9

STEP_SIZE = 1
GAMMA = 0.1

NUM_EPOCHS = 50
MODEL_SAVE_DIR = os.path.join(ROOT, "saved_model")

TRAIN_LOG_SAVE_PATH = os.path.join(ROOT, "train_log.txt")
LOG_EPOCH_MOD = 10
LOG_TIME_FORMAT = '%Y-%m-%d-%H-%M-%S----'

WRONG_IMG_DIR = os.path.join(ROOT, "predict_wrong")
