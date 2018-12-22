import os
from PIL import Image
from shutil import copyfile

import common

"""
Methods to setup the required environement for the project
"""

ENV = {
    "data" : {
        "train" : {
            "image" : {},
            "label" : {},
            "aug" : {},
        },
        "test" : {
            "image" : {},
            "foursplit" : {},
        },
        "split" : {
            "train" : {
                "image" : {},
                "label" : {},
            },
            "val" : {
                "image" : {},
                "label" : {},
            },
        },
    },
    "results" : {
        "train" : {
            "label" : {},
            "logits" : {},
            "overlay" : {},
        },
        "test" : {
            "label" : {},
            "logits" : {},
            "overlay" : {},
        },
        "logdir" : {},
    },
}

def get_paths(envdic, acc_path=""):
    """
    Lists paths of envdic leaves folders
    Used recursively on ENV, lists all leaves folders paths in ENV.
    Args:
        envdic: mapping dirname -> [envdic|{}], envdic if dirname has subfolders, {} if leaf folder (see ENV ex.)
        acc_path: accumulator path of current envdic for os.path.join
    Returns:
        list of full paths from acc_path_0 to the leaves directory (ie only those without subfolders)
    """
    return [acc_path] if not envdic else sum(
        (get_paths(subdic, os.path.join(acc_path, dirname)) for dirname, subdic in envdic.items()), [])

def complete_env(root_folder, *, verbose=False):
    """
    Completes ENV tree in root_folder
    Args:
        root_folder: path to needed root folder of ENV tree, if inexistant, is created.
    Raises:
        EnvironementError: when an existing directory that should be created is found on disk
    """
    vprint = common.GET_VERBOSE_PRINT(verbose)
    for d in get_paths(ENV, acc_path=root_folder):
        if os.path.isdir(d):
            continue
            #raise EnvironmentError(f"Found a pre-existing directory at {d}. Aborting.")
        vprint(f"[ENV] complete_env is making folder {d} because none pre-existing was found.")
        os.makedirs(d)

def check_env(root_folder, *, verbose=False):
    """
    Checks if the env at root_folder is complete
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations    
    Returns:
        The first missing folder path, or None in success case
    """
    vprint = common.GET_VERBOSE_PRINT(verbose)
    for d in get_paths(ENV, acc_path=root_folder):
        if not os.path.isdir(d):
            vprint(f"[ENV] Could not find subdirectory {d}.")
            return d
        vprint(f"[ENV] Found subdirectory {d}.")
    return None

def contains_all_test_images(root_folder, verbose=False):
    vprint = common.GET_VERBOSE_PRINT(verbose)
    test_img_files = os.listdir(os.path.join(root_folder, common.TEST_IMG_PATH))
    folder_full = len(test_img_files) >= 50
    vprint("{} {} test images.".format(common.TEST_IMG_PATH, "contains all" if folder_full else "is missing some"))
    return folder_full

def gen_four_split(original_images_dir, foursplit_dir):
    """
    Generate four splits images
    """
    fnames = os.listdir(original_images_dir)
    fnames.sort()
    for fn in fnames:
        if not "png" in fn: continue
        original_index = int(fn.replace("test_", '').replace(".png", ''))
        fpath = os.path.join(original_images_dir, fn)
        print(fpath)
        oim = Image.open(fpath)
        oim_name = os.path.basename(fpath)
        crops = [oim.crop(area) for area in ((0,0,400,400),(0,208,400,608),(208,0,608,400),(208,208,608,608))]
        for i, crop in enumerate(crops):
            imageid = "test_%.3d" % (4*(original_index-1) + i + 1)
            crop_save_path = os.path.join(foursplit_dir, f"{imageid}.png")
            print(i, original_index, fn, crop_save_path)
            crop.save(crop_save_path)

def prepare_train(root_folder, *, verbose=False):
    """
    Prepares the structure necessary for running utrain.py from the original dataset
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations   
    Raises:
        AssertionError: when the training/ original dataset is not found in root_folder
    """
    
    vprint = common.GET_VERBOSE_PRINT(verbose)

    missing = check_env(root_folder)
    if not missing:
        vprint(f"[ENV] No missing folder in the environement.")
        return
    vprint(f"[ENV] Some folders in the environment at {root_folder} are missing ({missing} first missing found). Recreating environment.")

    assert os.path.isdir(os.path.join(root_folder, "training/")), f"The training/ dataset folder was not found in {root_folder}."
    complete_env(root_folder, verbose=verbose)

    img_dir_path, gt_dir_path = (os.path.join(root_folder, "training", subf) for subf in ("images", "groundtruth"))
    vprint(f"[ENV] Using images and groundtruth folders {img_dir_path}, {gt_dir_path}")

    for fimg in os.listdir(img_dir_path):        
        fpath = os.path.join(img_dir_path, fimg)
        new_path = os.path.join(common.TRAIN_IMG_PATH, fimg)
        vprint(f"[ENV] Copying file {fpath} to {new_path}")
        copyfile(fpath, new_path)
        img_idx = int(fimg.split("_")[1].split(".")[0])
        new_split_path = os.path.join(
            root_folder,
            common.SPLIT_TRAIN_IMG_PATH if img_idx in common.SPLIT_TRAIN_INDICES else common.SPLIT_VAL_IMG_PATH,
            fimg)
        vprint(f"[ENV] Copying file {fpath} to {new_split_path}")
        copyfile(fpath, new_split_path)

    for fgt in os.listdir(gt_dir_path):        
        fpath = os.path.join(gt_dir_path, fgt)
        new_path = os.path.join(common.TRAIN_GT_PATH, fgt)
        vprint(f"[ENV] Copying file {fpath} to {new_path}")
        copyfile(fpath, new_path)
        img_idx = int(fgt.split("_")[1].split(".")[0])
        new_split_path = os.path.join(
            root_folder,
            common.SPLIT_TRAIN_GT_PATH if img_idx in common.SPLIT_TRAIN_INDICES else common.SPLIT_VAL_GT_PATH,
            fgt)
        vprint(f"[ENV] Copying file {fpath} to {new_split_path}")
        copyfile(fpath, new_split_path)
    
def prepare_test(root_folder, *, verbose=False):
    """
    Prepares the structure necessary for running utrain.py from the original dataset
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations   
    Raises:
        AssertionError: when the training/ original dataset is not found in root_folder
    """

    vprint = common.GET_VERBOSE_PRINT(verbose)

    missing = check_env(root_folder)
    vprint(f"[ENV] check_env on {root_folder} returned {missing}")
    if not missing:
        contains_all_imgs = contains_all_test_images(root_folder, verbose=True)
        vprint(f"[ENV] contains_all_test_images returned {contains_all_imgs}")
        if contains_all_imgs:
            return
        #raise Exception("[ENV] env is complete but all the images are not there. Delete ENV and restart.")
    else:
        complete_env(root_folder, verbose=verbose)
    test_img_folder = os.path.join(common.TEST_PATH, common.IMG_SUBFOLDER)
    test_img_folders_root = os.path.join(root_folder, "test_set_images/") 
    assert os.path.isdir(test_img_folders_root), f"The test_set_images/ dataset folder was not found in {root_folder}."
    

    for original_test_img_folder in os.listdir(test_img_folders_root):
        original_test_img_folder_path = os.path.join(test_img_folders_root, original_test_img_folder)
        for img in os.listdir(original_test_img_folder_path):
            fpath = os.path.join(original_test_img_folder_path, img)
            imgidx = int(img.split("_")[1].split(".")[0])
            new_fname = "test_%.3d.png" % imgidx
            new_path = os.path.join(test_img_folder, new_fname)
            vprint(f"[ENV] Moving file {fpath} to {new_path}")
            copyfile(fpath, new_path)

    gen_four_split(test_img_folder, os.path.join(root_folder, common.TESTING_PATH_FOURSPLIT))
