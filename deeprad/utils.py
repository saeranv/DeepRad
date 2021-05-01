"""Utility functions for deeprad."""
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except:
    cv2 = None
    print("Skip cv2 installation.")

from typing import List
import shapely.geometry as geom
from pprint import pprint

# Path to all models in deep_rad
DEEPRAD_MODELS_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/models/'))
DEEPRAD_GHOUT_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/ghout/'))
DEEPRAD_TRAINTEST_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/traintest/'))
RADCMAP = plt.get_cmap('RdYlBu_r')

def pp(x, *args):
    pprint(x) if not args else print(x, *args)

def timer(start_time, end_time):
  hours, rem = np.divmod(end_time-start_time, 3600)
  minutes, seconds = np.divmod(rem, 60)
  return "h:{} m:{} s:{}".format(int(hours), int(minutes), int(seconds))

def fd(module, key=None):
    """ To efficiently search modules."""
    def hfd(m, k): return k.lower() in m.lower()
    if key is None:
        return [m for m in dir(module)][::-1]
    else:
        return [m for m in dir(module) if hfd(m, key)][::-1]


def to_poly_sh(xy_arr):
    """Shapely polygon from list of two arrays of x, y coordinates.

        Args:
            xy_arr: [x_arr, y_arr]
    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return geom.Polygon([(x, y) for x, y in zip(*xy_arr)])


def to_poly_np(poly_sh):
    """Two arrays of x, y coordinates from shapely polygon.

    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return np.array(poly_sh.exterior.xy)


def load_img_gray(img_fpath: str) -> np.ndarray:
    return cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)


def load_img_rgb(img_fpath: str) -> np.ndarray:
    return cv2.imread(img_fpath, cv2.COLOR_BGR2RGB)


def write_img(img: np.ndarray, img_fpath: str) -> bool:
    return cv2.imwrite(img_fpath, img)

def color2rad(img, mask=False):
    """img is np.ndarray of floats btwn 0 - 1"""
    img = (img * 255).astype(np.uint8)
    # TODO: add a mask here??
    if mask:
        img = np.where(img < (255 - 1e-10), img, np.nan)
    return img

def load_json(json_fpath):
    with open(json_fpath, 'r') as fp:
        val_dict = json.load(fp)

    return val_dict

# TODO: these two functions don't belong in utils
def extract_floorplan_ids(data_num, target_data_dir=None, verbose=True):
    """Safely extract root model directories for polygon extraction."""

    if target_data_dir is None:
        target_data_dir = DEEPRAD_MODELS_DIR

    # Load all model directories
    floorplan_id_arr = os.listdir(target_data_dir)
    floorpla_id_arr = sorted(floorplan_id_arr)
    n_ids = len(floorplan_id_arr)

    if n_ids == 0:
        raise Exception('No image files. Add images and try again.')

    if data_num > n_ids:
        if verbose:
            print('Note: data_num {} is too high. Resetting to n_ids {}.'.format(
                  data_num, n_ids))
        data_num = n_ids

    return data_num, floorplan_id_arr


def load_floorplan_data(targ_id_dirs, data_num):
    """Load floorplan data."""

    src_img_arr = [0] * data_num
    label_img_arr = [0] * data_num
    hdict_arr = [0] * data_num
    targ_id_dir_arr = [0] * data_num

    i = -1
    idx = -1
    total_i = 0
    null_lst = []

    while (idx + 1) < data_num:
        i += 1
        targ_id_dir = os.path.join(
            DEEPRAD_MODELS_DIR, targ_id_dirs[i], 'data')
        targ_src_fpath = os.path.join(targ_id_dir, 'src.jpg')
        targ_label_fpath = os.path.join(targ_id_dir, 'label.jpg')
        targ_json_fpath = os.path.join(targ_id_dir, 'door_vecs.json')

        hdict = load_json(targ_json_fpath)

        # if hdict['scale'] < 0.4:
        #     print('Skip {} b/c scale at {}. Total skipped={}.'.format(
        #           targ_id_dirs[i], hdict['scale'], total_i))
        #     total_i += 1
        #     continue

        idx += 1
        hdict_arr[idx] = hdict
        src_img_arr[idx] = load_img_rgb(targ_src_fpath)
        label_img_arr[idx] = load_img_gray(targ_label_fpath)
        targ_id_dir_arr[idx] = targ_id_dir
        null_lst.append(targ_id_dirs[i] + '\n')

    # Write to null list
    null_fpath = os.path.join(DEEPRAD_MODELS_DIR, '_null.txt')
    with open(null_fpath, 'w') as fp:
        # [fp.writeline(null_) for null_ in null_lst]
        fp.writelines(null_lst)
    return hdict_arr, src_img_arr, label_img_arr, targ_id_dir_arr


def make_dir_safely(dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)


def fname_from_fpath(fpath):
    """Splits filename from extension and preceding directories."""
    return Path(fpath).stem


def to_multi_channel_img_arr(imgs: List[np.ndarray]):
    """Safely constructs a multichannel array from multiple grayscale images."""

    n_channels = len(imgs)

    assert np.all([np.array_equal([len(img.shape)], [2])
                   for img in imgs]), 'img must be 2d grayscale.'

    assert np.all([np.array_equal(img.shape, imgs[0].shape) for img in imgs]), \
        'All images must be same shape. Got {}.'.format(
            [img.shape for img in imgs])

    cat_img = np.dstack(imgs)  # Concat depthwise channels
    assert cat_img.shape[2] == n_channels

    return cat_img
