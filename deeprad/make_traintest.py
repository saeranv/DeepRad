"""Preprocess raw outputs from GHOUT_DIR for train/test dataset."""
import sys
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


def make_channel_set(floorplan_id) -> set:
    """Creates dictionary of unqiue channels from input directory."""

    in_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'in_data')
    in_fpaths = [os.path.join(in_dir, _in) for _in in os.listdir(in_dir)]

    chset = set()
    for fpath in in_fpaths:
        fname = utils.fname_from_fpath(fpath)
        chtype = fname.split('_')[0]  # remove orientation info
        if chtype not in chset:
            chset.add(chtype)

    return chset


def preprocess_traintest_out(floorplan_id: str) -> None:
    """Preprocess traintest_out"""

    # Get directories
    out_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'out_label')

    label_fpaths = [os.path.join(out_dir, _out)
                    for _out in os.listdir(out_dir)]
    label_imgs = [utils.load_img_gray(f) for f in label_fpaths]
    out_img = preprocess_img(label_imgs)
    out_img_fpath = os.path.join(
        DEEPRAD_TRAINTEST_DIR, 'out_data', "out_{}.jpg".format(floorplan_id))
    assert utils.write_img(out_img, out_img_fpath)


def preprocess_traintest_in(floorplan_id: str, chset: set) -> None:
    """Preprocess traintest_in"""

    in_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'in_data')
    in_fpaths = [os.path.join(in_dir, _in) for _in in os.listdir(in_dir)]

    # Concat multiple images into one image depthwise (as chennels)
    for j, ch in enumerate(chset):
        ch_fpaths = [in_fpath for in_fpath in in_fpaths if ch in in_fpath]
        _ch_imgs = [utils.load_img_gray(ch_fpath)
                    for ch_fpath in ch_fpaths]
        _ch_img = preprocess_img(_ch_imgs)  # concats across xdim
        in_img_fpath = os.path.join(
            DEEPRAD_TRAINTEST_DIR, 'in_data', "in_{}_{}.jpg".format(ch, floorplan_id))
        assert utils.write_img(_ch_img, in_img_fpath), 'Image {} failed to save ' \
            'in_img_fpath'.format(in_img_fpath)

    return chset


def main(verbose=True, nuke_all=True):
    """Preproces main."""

    # load all directories in train/test
    data_num, floorplan_ids = \
        utils.extract_floorplan_ids(1e6, DEEPRAD_GHOUT_DIR, verbose=False)

    if nuke_all:
        in_old_imgs = os.listdir(os.path.join(
            DEEPRAD_TRAINTEST_DIR, 'in_data'))
        out_old_imgs = os.listdir(os.path.join(
            DEEPRAD_TRAINTEST_DIR, 'out_data'))

        [os.remove(os.path.join(DEEPRAD_TRAINTEST_DIR, 'in_data', old_img))
         for old_img in in_old_imgs]
        [os.remove(os.path.join(DEEPRAD_TRAINTEST_DIR, 'out_data', old_img))
         for old_img in out_old_imgs]

    chset = make_channel_set(floorplan_ids[0])
    n_floorplans = data_num

    for i, floorplan_id in enumerate(floorplan_ids):

        if verbose:
            print('{}/{} Preprocessing {}.'.format(i +
                  1, n_floorplans, floorplan_id))

        preprocess_traintest_out(floorplan_id)
        preprocess_traintest_in(floorplan_id, chset)


if __name__ == "__main__":

    nuke_all = True
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        if '--nuke_all' in argv:
            i = argv.index('--nuke_all')
            nuke_all = bool(argv[i + 1])

    main(nuke_all=nuke_all)
