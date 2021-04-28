"""Preprocess raw outputs from GHOUT_DIR for train/test dataset."""
import sys
import os
import numpy as np
from typing import List

# import cv2

import deeprad.utils as utils
import deeprad.image2vec as i2v

DEEPRAD_GHOUT_DIR = utils.DEEPRAD_GHOUT_DIR
DEEPRAD_TRAINTEST_DIR = utils.DEEPRAD_TRAINTEST_DIR
MAX_CROP_SIZE = 0


def crop_orients(img, border_width=2, base_x=1300):
    """Crop x, y dims. border_width is edges of images."""
    _img = img.copy()
    ret_mid_xdim = int(base_x / 2)
    gap_tol = 100  # naturally occuring gaps in buildings in pixels
    assert border_width % 2 == 0, 'border_width must be multiple of 2.'
    buf2 = border_width
    ret_img = np.ones((110, ret_mid_xdim * 2)) * 255

    # create bit x-proj
    bit_img = np.where(img == 255, 0, 1)
    xproj = np.sum(bit_img, axis=0).astype(bool)
    xproj = np.logical_not(xproj)

    # xproj_viz = np.stack([xproj] * 300, axis=0)
    # plt.imshow(xproj_viz, cmap='gray')

    white_idx = i2v.contiguous_ones_idx(xproj)
    pix_gap = white_idx[:, 1] - white_idx[:, 0]

    gap_idx = np.where(pix_gap > gap_tol)
    white_idx = white_idx[gap_idx]
    # print(white_idx)

    crop_img = [0] * (white_idx.shape[0] - 1)
    ymin, ymax = 50, 150 + 10
    buf = int(border_width / 2)
    # crop from original image
    for i in range(white_idx.shape[0] - 1):
        i1, i2 = white_idx[i, 1], white_idx[i + 1, 0]
        # TODO: add a border
        _crop_img = _img[ymin:ymax, i1 - buf2:i2 + buf2]
        # print(i1-buf2,i1-buf)
        # _crop_img[:,i1-buf2:i1-buf] = 0
        # _crop_img[:, :200 ] = 0
        # _crop_img[:,i2+buf:i2+buf2] = 0
        crop_img[i] = _crop_img

    crop_img = np.concatenate(crop_img, axis=1)

    if crop_img.shape[1] > ret_img.shape[1]:
        print('crop_img larger then return img. Expand overall size '
              ' to {} from {}.'.format(crop_img.shape, ret_img.shape))
        diff = int(crop_img.shape[1] - ret_img.shape[1] / 2.0)
        crop_img = crop_img[:, diff + 1:crop_img.shape[1] - diff - 1]
        if crop_img.shape[1] > MAX_CROP_SIZE:
            MAX_CROP_SIZE = crop_img.shape[1]

    # center add to ret_img
    mid_xdim = int(np.floor(crop_img.shape[1] / 2.0))
    move_idx = ret_mid_xdim - mid_xdim
    ret_img[:, move_idx: move_idx + crop_img.shape[1]] = crop_img

    return ret_img


def preprocess_img(img_lst: List[np.ndarray], downsample: int = 1) -> np.ndarray:
    """Preprocess image for traintest."""

    ret_img = None

    # contiguous ones and trim.
    # concatenate all images acrosss x-axis
    ret_img = np.concatenate(img_lst, axis=1)

    # TODO: Swap this so you don't have to guess gap tol
    ret_img = crop_orients(ret_img)

    # check dim
    #xdim = img_lst[0].shape[1]
    #assert ret_img.shape[1] == (xdim * len(img_lst))

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


def preprocess_traintest_out(floorplan_id: str, idx: int) -> None:
    """Preprocess traintest_out"""

    # Get directories
    out_dir = os.path.join(DEEPRAD_GHOUT_DIR, floorplan_id, 'out_label')

    label_fpaths = [os.path.join(out_dir, _out)
                    for _out in os.listdir(out_dir)]

    label_imgs = [utils.load_img_gray(f) for f in label_fpaths]
    out_img = preprocess_img(label_imgs)
    out_img_fpath = os.path.join(
        DEEPRAD_TRAINTEST_DIR, 'out_data', "{}_out_{}.jpg".format(idx, floorplan_id))

    assert utils.write_img(out_img, out_img_fpath)


def preprocess_traintest_in(floorplan_id: str, chset: set, idx: int) -> None:
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
            DEEPRAD_TRAINTEST_DIR, 'in_data', "{}_in_{}_{}.jpg".format(
                idx, ch, floorplan_id))
        assert utils.write_img(_ch_img, in_img_fpath), 'Image {} failed to save ' \
            'in_img_fpath'.format(in_img_fpath)

    return chset


def main(verbose=True, data_num=None, nuke_all=True):
    """Preproces main."""

    if data_num is None:
        data_num = 1e6
    # load all directories in train/test
    data_num, floorplan_ids = \
        utils.extract_floorplan_ids(data_num, DEEPRAD_GHOUT_DIR, verbose=False)

    floorplan_ids = floorplan_ids[:data_num]
    # print(data_num)
    # print(floorplan_ids[:])

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
    comp = 0
    for i, floorplan_id in enumerate(floorplan_ids):

        if verbose:
            print('{}/{} Preprocessing {}.'.format(
                i + 1, n_floorplans, floorplan_id))
        try:
            preprocess_traintest_out(floorplan_id, i - comp)
            preprocess_traintest_in(floorplan_id, chset, i - comp)
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print('error @ {} {}'.format(i, floorplan_id))
            comp += 1
    print('max size', MAX_CROP_SIZE)


if __name__ == "__main__":

    nuke_all, data_num = True, None
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        if '--nuke_all' in argv:
            i = argv.index('--nuke_all')
            nuke_all = bool(argv[i + 1])
            print('nuke_all={}'.format(nuke_all))
        if '--data_num' in argv:
            i = argv.index('--data_num')
            data_num = int(argv[i + 1])
            print('data_num={}'.format(data_num))

    main(data_num=data_num, nuke_all=nuke_all)
