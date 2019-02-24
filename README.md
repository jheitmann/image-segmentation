
Machine Learning Project 2
=====

Julien Niklas Heitmann  
Philipp Khlebnikov  
Louis Gabriel Jean Landelle

This project uses a U-Net architecture to predict a segmentation problem on a small dataset of satellite images, segmenting roads from non-roads areas. It is built with Keras on a TensorFlow back-end. Our best model yields a F1 score of 0.888 on the testing dataset. This README describes the steps to reproduce our results, and a general project structure overview.

## Preparation

This project was made using Python 3.6.5 with packages TensorFlow 1.10.0 and Keras 2.2.4
We also use numpy, matplotlib, PIL, cv2.
In order to reproduce our results, you have to perform those preparation steps, in some ROOT_FOLDER:

- Unzip the datasets in ROOT_FOLDER (in order to have test_set_images/ and training/)

- Place all our .py scripts in ROOT_FOLDER

## Training and using the best model

You can just use run.py to automatically perform the training and testing steps using our best model.

This script executes those steps:

- Launches utrain.py to retrain our model on the training dataset. This performs:
    - creating all of the environment empty folders following our structure
    - transfering training/ images to data/train/image/, groundtruth to data/train/label/
    - executing training, and generating *.hdf5 checkpoints in results/, named to represent the chosen parameters, whenever an improvement is made during the chosen number of steps
    - because we removed all sources of randomness, the same *.hdf5 as ours should result from this script

- Launches utest.py on this best checkpoint. This performs:
    - transfering test_set_images/test_*/*.png to data/test/image/
    - executing testing and generating the resulting masks
    - generating a submission csv in results/ ready for upload to CrowdAI
    - because we removed all sources of randomness, the F1 score should be the same
     
In case randomness makes the training stagnate, first try with different seed in common.py, else download weights at https://uploadfiles.io/xcpjk

## More options

We prepared a notebook, experiments.ipynb, representing the many tries we performed over the course of each incremental improvement. Each part of the notebook launches training directly from utrain.py using some combination of parameters, and logs the metrics to a csv file, which is then converted to a graph image. If needed, you should be able to execute each cell and see the results for yourself. Because the random SEED in common.py was changed many times, the performances may vary as a result, but should be similar to those described in the report.

## Code structure

This project code is structured as follows:
- common.py : contains all of our constants, most of them used throughout the other scripts
- model.py : the keras model of the U-Net neural network
- preprocessing.py : functions used for dataset extraction and preparation
- postprocessing.py : functions used for results images, csvs, generation, aggregation, etc.
- setup_env.py : functions used for automatic creation and checking of the folders needed
- utrain.py : script executing the logic for training, which can either be imported and used via its main() function or directly launched in terminal (see help via python utrain.py -h). This script is responsible for the generation of the hdf5 checkpoint files, and the csv/graph images logging the metrics evolutions
- utest.py : script executing the logic for testing, which can either be imported and used via its main() function or directly launched in terminal (see help via python utest.py -h). This script is responsible for the generation of the masks images, optionally the logits and the overlay files, and the csv submission files.

## Usage of the utrain and utest scripts

The lowest public abstraction level of our code is the direct usage of utrain.py and utest.py scripts via a terminal. This is how we generated most of our results. Calling utrain and utest requires parameters, which are described in the help of the argparse (python utrain.py -h, python utest.py -h). For example, this is how we would train a U-Net with input size (256\*256\*3), using dataset augmentation and our chosen validation dataset, with a batch size of 2, during 200 epochs of 80/20 training/validation steps each:

>python utrain.py 256 2 200 80 -rgb -aug -cv 

and after the hdf5 is generated in results/, this is how we would predict the testing dataset to generate a csv submission file:

>python utest.py results/unet_rgb_256_2018-12-19_15_30_10_xxxxxx.hdf5

(this is assuming the training is launched on 2018-12-19 at 15:30:10)
