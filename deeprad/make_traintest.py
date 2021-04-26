"""Preprocess raw outputs from GHOUT_DIR for train/test dataset."""
import os
import numpy as np
from typing import List

import cv2

import deeprad.utils as utils
import deeprad.image2vec as i2v

DEEPRAD_GHOUT_DIR = utils.DEEPRAD_GHOUT_DIR
DEEPRAD_TRAINTEST_DIR = utils.DEEPRAD_TRAINTEST_DIR


def preprocess_img(img_lst: List[np.ndarray], downsample: int = 1) -> np.ndarray:
    """Preprocess image for traintest."""

    ret_img = None
    # TODO: contiguous ones and trim.
    # concatenate all images acrosss x-axis

    ret_img = np.concatenate(img_lst, axis=1)

    # check dim
    xdim = img_lst[0].shape[1]
    assert ret_img.shape[1] == (xdim * len(img_lst))

    # downsample
    if downsample > 1:
        ret_img = ret_img[::downsample, ::downsample]

    return ret_img


def in_channel_set(in_fpaths: List[str]) -> dict:
    """Creates dictionary of unqiue channels from input directory."""

    chset = set()
    for fpath in in_fpaths:
        fname = utils.fname_from_fpath(fpath)
        chtype = fname.split('_')[0]  # remove orientation info
        if chtype not in chset:
            chset.add(chtype)

    return chset


def main():
    """Preproces main."""

    # load all directories in train/test
    data_num, floorplan_ids = \
        utils.extract_floorplan_ids(1e6, DEEPRAD_GHOUT_DIR, verbose=False)

    chset = set()

    # for i, floorplan_id in enumerate(floorplan_ids):
    i = 0
    floorplan_id = floorplan_ids[0]

    # Make filepaths and directores
    in_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'in_data')
    out_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'out_label')
    in_img_fpath = os.path.join(
        DEEPRAD_TRAINTEST_DIR, "in_{}.jpg".format(floorplan_id))
    out_img_fpath = os.path.join(
        DEEPRAD_TRAINTEST_DIR, "out_{}.jpg".format(floorplan_id))

    # # Preprocess traintest_out
    # label_fpaths = [os.path.join(out_dir, _out)
    #                 for _out in os.listdir(out_dir)]
    # label_imgs = [utils.load_img_gray(f) for f in label_fpaths]
    # out_img = preprocess_img(label_imgs)
    # assert utils.write_img(out_img, out_img_fpath)

    # Preprocess traintest_in
    in_fpaths = [os.path.join(in_dir, _in) for _in in os.listdir(in_dir)]
    in_fpaths = [os.path.join(in_dir, _in) for _in in os.listdir(in_dir)]
    if len(chset) == 0:
        chset = in_channel_set(in_fpaths)

    # Concat multiple images into one image depthwise (as chennels)
    n_channels = len(chset)  # number of channels
    ch_imgs = [0] * n_channels
    for j, ch in enumerate(chset):
        ch_fpaths = [in_fpath for in_fpath in in_fpaths if ch in in_fpath]
        _ch_imgs = [utils.load_img_gray(ch_fpath) for ch_fpath in ch_fpaths]
        ch_imgs[j] = preprocess_img(_ch_imgs)  # concats across xdim
    print(floorplan_id)
    # print(ch_imgs)
    in_img = utils.to_multi_channel_img(ch_imgs)
    assert utils.write_img(in_img, in_img_fpath)


if __name__ == "__main__":
    main()
