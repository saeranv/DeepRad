"""Utility functions for deeprad."""
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Path to all models in deep_rad
DEEPRAD_MODELS_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/models/'))
DEEPRAD_GHOUT_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/ghout/'))
DEEPRAD_TRAINTEST_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/traintest/'))


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


def extract_floorplan_ids(data_num, target_data_dir=None, verbose=True):
    """Safely extract root model directories for polygon extraction."""

    if target_data_dir is None:
        target_data_dir = DEEPRAD_MODELS_DIR

    # Load all model directories
    floorplan_id_arr = os.listdir(target_data_dir)
    n_ids = len(floorplan_id_arr)

    if n_ids == 0:
        raise Exception('No image files. Add images and try again.')

    if data_num > n_ids:
        if verbose:
            print('Note: data_num {} is too high. Resetting to n_ids {}.'.format(
                  data_num, n_ids))
        data_num = n_ids

    return data_num, floorplan_id_arr


def load_json(json_fpath):
    with open(json_fpath, 'r') as fp:
        val_dict = json.load(fp)

    return val_dict


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

        if hdict['scale'] < 0.4:
            print('Skip {} b/c scale at {}. Total skipped={}.'.format(
                  targ_id_dirs[i], hdict['scale'], total_i))
            total_i += 1
            continue

        idx += 1
        hdict_arr[idx] = hdict
        src_img_arr[idx] = koad_img_rgb(targ_src_fpath)
        label_img_arr[idx] = load_img_gray(targ_label_fpath)
        targ_id_dir_arr[idx] = targ_id_dir
        null_lst.append(targ_id_dirs[i] + '\n')

    # Write to null list
    null_fpath = os.path.join(DEEPRAD_MODELS_DIR, '_null.txt')
    with open(null_fpath, 'w') as fp:
        #[fp.writeline(null_) for null_ in null_lst]
        fp.writelines(null_lst)
    return hdict_arr, src_img_arr, label_img_arr, targ_id_dir_arr


def make_dir_safely(dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
