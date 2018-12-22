import argparse
import numpy as np
import os

import common
from model import unet
from postprocessing import four_split_mean, predictions_to_masks, masks_to_submission, compute_trainset_f1
from preprocessing import extract_data
from setup_env import check_env, prepare_test, prepare_train, gen_four_split


def main(ckpt_path, four_split, use_max, training, *, foreground_threshold=0.25, root_folder=None):
    """
    Args:
        ckpt_path: path of model Checkpoint
        t: bool set True if training and not testing set should be predicted
        four_split: bool set True if four predictions should be computed per test image, and then combined
        foreground_threshold: threshold used to determine when to label a path as 'foreground'
        root_folder: use to override root_folder=os.getcwd (Typically when using main() in Google Colab)
    """
    # Prepares environment for the testing (creates the folder structure needed for our code)
    prepare_test(os.getcwd() if not root_folder else root_folder, verbose=True)

    # Extracts testing parameters from the filename of the hdf5 file
    img_height = int(os.path.basename(ckpt_path).split("_")[2].split('.')[0])
    rgb = "rgb" in ckpt_path
    n_channels = 3 if rgb else 1    
    input_size = (img_height,img_height,n_channels)
    # Recreates the model using the weights of the checkpoint
    model = unet(input_size, pretrained_weights=ckpt_path)

    print('Neural network input size:', input_size)

    # Extracts the testing dataset, depending on if we use the four_split method or not
    if four_split:
        assert not training, "Four split on training dataset is a bad idea (images already at correct scale)"
        imgs = extract_data(common.TESTING_PATH_FOURSPLIT, common.N_TEST_IMAGES * 4, img_height, rgb)
    else:
        if training:
            imgs = extract_data(common.TRAIN_IMG_PATH, common.N_TRAIN_IMAGES, img_height, rgb)
        else:
            imgs = extract_data(common.TEST_IMG_PATH, common.N_TEST_IMAGES, img_height, rgb)
    
    # Predicts the testing dataset using the model
    print("Computing predictions...")
    preds = model.predict(imgs, batch_size=1, verbose=1)
    
    # Generates prediction masks
    print('Generating predicted masks in', common.RESULTS_PATH)
    result_path = common.RESULTS_PATH + ("train/" if training else "test/")
    test_name = os.path.join(common.TRAIN_IMG_PATH, "satImage") if training else os.path.join(common.TEST_IMG_PATH, "test")
    output_height = common.TRAIN_IMG_HEIGHT if training or four_split else common.TEST_IMG_HEIGHT
    predicted_mask_files = predictions_to_masks(result_path, test_name, preds, output_height, 
                                                    four_split, common.TEST_IMG_HEIGHT, use_max)
    
    # Generates submission csv
    print('Generating submission at', common.SUBM_PATH)
    masks_to_submission(common.SUBM_PATH, predicted_mask_files, foreground_threshold=foreground_threshold)

    # Displays the F1_score, when we use training dataset validation
    if training:
        f1_score = compute_trainset_f1(common.SUBM_PATH)
        print(f"Training set f1-score: {f1_score}")

    return common.SUBM_PATH


if __name__=="__main__":
    # Defines all parser arguments when launching the script directly in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str,
                        help="path to ckpt file")
    parser.add_argument("-fs", "--four_split", help="take mean of four predictions on testing images",
                        action="store_true")
    parser.add_argument("-max", "--use_max", help="use max instead of mean to combine the four predictions",
                        action="store_true")
    parser.add_argument("-t", "--training", help="predict training set instead of testing",
                        action="store_true")
            
    args = parser.parse_args()
    main(args.ckpt_path, args.four_split, args.use_max, args.training)
