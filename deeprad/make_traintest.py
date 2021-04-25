"""Preprocess raw outputs from GHOUT_DIR for train/test dataset."""
import os
import numpy as np

import cv2

import deeprad.utils as utils
import deeprad.image2vec as i2v

DEEPRAD_GHOUT_DIR = utils.DEEPRAD_GHOUT_DIR
DEEPRAD_TRAINTEST_DIR = utils.DEEPRAD_TRAINTEST_DIR


def preprocess_img(img_fpath: str) -> np.ndarray:
    """Preprocess image for traintest."""

    print(img_fpath)


def main():
    """Preproces main."""

    # load all directories in train/test
    data_num, floorplan_ids = \
        utils.extract_floorplan_ids(1e6, DEEPRAD_TRAINTEST_DIR, verbose=False)

    # for i, floorplan_id in enumerate(floorplan_ids):
    i = 0
    floorplan_id = floorplan_ids[0]

    in_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'in_data')
    out_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'out_label')

    # Preprocess labels
    label_fpaths = [os.path.join(out_dir, out) for out in os.listdir(out_dir)]
    label_imgs = [preprocess_img(fpath) for fpath in label_fpaths]
    #[save_img(img) for img in label_imgs]

    # Preprocess input images
    # add as channels

    # save new images

    pass


if __name__ == "__main__":
    main()
