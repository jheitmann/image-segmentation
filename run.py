import os
import common
import utrain
import utest

N_EPOCHS = 1000
N_STEPS_PER_EPOCH = 80

dgargs = dict(    
    rotation_range=90,
    fill_mode='reflect',
    horizontal_flip=True,
    vertical_flip=True
)
file_id = utrain.main(
    img_height=256,
    batch_size=2,
    epochs=N_EPOCHS,
    steps_per_epoch=N_STEPS_PER_EPOCH,
    aug=True,
    chosen_validation=True,
    rgb=False,
    pretrained_weights=None,
    monitor="val_acc",
    root_folder=".",
    data_gen_args=dgargs
)
hdf5_path = os.path.join(common.RESULTS_PATH, file_id + ".hdf5")
subm_path = utest.main(
    ckpt_path=hdf5_path,
    use_max=True,
    four_split=True,
    training=False,
)
print(f"Saved submission file at {subm_path}.")
