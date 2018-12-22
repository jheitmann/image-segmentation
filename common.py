import os

SEED = 2018 # seed used for randomness throughout the project

#utrain
SPLIT_TRAIN_PATH = "data/split/train"
SPLIT_VAL_PATH = "data/split/val"
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
IMG_SUBFOLDER = "image"
GT_SUBFOLDER = "label"

N_SPLIT_TRAIN = 80
N_SPLIT_VAL = 20
N_TRAIN_IMAGES = 100
AUG_SAVE_PATH = "data/train/aug/"

TRAIN_IMG_PATH = os.path.join(TRAIN_PATH, IMG_SUBFOLDER)
TRAIN_GT_PATH = os.path.join(TRAIN_PATH, GT_SUBFOLDER)

SPLIT_TRAIN_IMG_PATH = os.path.join(SPLIT_TRAIN_PATH, IMG_SUBFOLDER)
SPLIT_TRAIN_GT_PATH = os.path.join(SPLIT_TRAIN_PATH, GT_SUBFOLDER)
SPLIT_VAL_IMG_PATH = os.path.join(SPLIT_VAL_PATH, IMG_SUBFOLDER)
SPLIT_VAL_GT_PATH = os.path.join(SPLIT_VAL_PATH, GT_SUBFOLDER)

SPLIT_VALID_INDICES = [26, 33, 65, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99, 100]
SPLIT_TRAIN_INDICES = [idx for idx in range(1, N_TRAIN_IMAGES) if not idx in SPLIT_VALID_INDICES]

DEFAULT_GEN_ARGS = dict(
    rotation_range=90,
    fill_mode='reflect',
    horizontal_flip=True,
    vertical_flip=True)

#utest
TEST_IMG_PATH = os.path.join(TEST_PATH, IMG_SUBFOLDER)

TESTING_PATH_FOURSPLIT = "data/test/foursplit/"
RESULTS_PATH = "results/"
SUBM_PATH = "results/output.csv"
LOGDIR = "results/logdir"
N_TEST_IMAGES = 50
TEST_IMG_HEIGHT = 608
TRAIN_IMG_HEIGHT = 400

#pre/postprocessing
PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16
PIXEL_THRESHOLD = 127
PREDS_PER_IMAGE = 4
AREAS = ((0,0,400,400),(208,0,608,400),(0,208,400,608),(208,208,608,608))

# returns print if verbose==True, otherwise an invisible function w. same signature
GET_VERBOSE_PRINT = lambda verbose: (lambda *a, **kwa: print(*a, **kwa) if verbose else None)