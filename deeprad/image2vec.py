
"""Generate vector data from floorplan images."""
import sys
import os
import json
from copy import copy

# scientific python stuff
import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
import shapely.geometry as geom
from rtree import index
from sklearn.mixture import GaussianMixture
from matplotlib.axes import Axes

# Path to all models in deep_rad
DEEPRAD_DATA_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/models/'))


def viz_poly_sh_arr(poly_sh_arr, a=None, scale=1, iter=False, **kwargs):
    """Plot shapely polygons"""

    n_plots = 1
    if iter:
        n_plots = len(poly_sh_arr)
    if a is None:
        _, a = plt.subplots(1, n_plots, sharey=True)

    for i, poly_sh in enumerate(poly_sh_arr):
        if poly_sh.exterior:
            xx, yy = poly_sh.exterior.xy
            xx, yy = np.array(xx), np.array(yy)
            if iter:
                a[i].plot(xx * scale, yy * scale, **kwargs)
            else:
                a.plot(xx * scale, yy * scale, **kwargs)
    if iter:
        a[i].axis('equal')
        return a[i]
    else:
        a.axis('equal')
        return a


def viz_ortho_snap_grid(x_means: np.ndarray, y_means: np.ndarray, ax: Axes, color: str = 'red',
                        linewidth: int = 2, alpha: float = 0.5, **kwargs) -> Axes:
    """Visualize polygon ortho snap grid."""

    x_min, x_max = x_means.min(), x_means.max()
    y_min, y_max = y_means.min(), y_means.max()

    # TODO: fix kwargs
    for x_mean, y_mean in zip(x_means, y_means):
        ax.plot([x_mean, x_mean], [y_min, y_max],
                color='red', linewidth=2, alpha=0.5, **kwargs)
        ax.plot([x_min, x_max], [y_mean, y_mean],
                color='red', linewidth=2, alpha=0.5, **kwargs)

    return ax


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


def rooms_equal_color_mapping(_rooms: np.ndarray) -> np.ndarray:
    """Replace class indices with equally spaced intervals from 0-255 for thresholding.

        Args:
            _rooms: 2d matrix of room image pixels in np.uint8.

        Returns:
            _rooms: 2d matrix of room image pixels with indices
                mapped between 0-255.

    """

    if (len(_rooms.shape) != 2) or (not _rooms.dtype == np.uint8):
        raise Exception('_rooms expects 2D matrix of dtype np.uint.'
                        'Got: {} shape and {} dtype.'.format(_rooms.shape, _rooms.dtype))

    room_classes = np.unique(_rooms)
    N = room_classes.size

    # Color val calculation
    color_interval = 255 / (N - 1)
    color_vals = (np.arange(N) * color_interval).astype(np.uint8)
    # color_vals = (np.ones(N) * 50).astype(np.uint8)

    # Replace room indices with colors
    for i, v in zip(room_classes, color_vals):
        _rooms = np.where(_rooms == i, v, _rooms)

    return _rooms


def contour_to_poly_np(pt):
    """opencv to polygon array."""
    pt = pt.T[:, 0]
    x, y = pt
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])
    return np.array([x, y])


def contour_arr_from_bit_image(_img, threshold=False):
    """Get opencv contours from binary (black/white) raster image."""

    # Optional ways to preprocess image for thresholding.
    if threshold:
        block_size = 5  # size of pixel neighborhood for threshold
        bias = 0  # subtract from mean
        _thresh = cv2.adaptiveThreshold(
            _img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, bias)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        _img = cv2.dilate(_thresh, kernel).copy()
    _, _contours, hier = cv2.findContours(_img, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    # hier = hier[0]
    return _contours


def clean_poly_sh(poly_sh, buffer_epsilon, simplify_tol, convex_diff_pct=0.1):
    """Applies methods to clean shapely geometry.

    Simplfiies w/ tolerance, buffers, and checks ccw orientation.
    """

    # Buffer in/out to remove some geometric flaws
    poly_sh = poly_sh.buffer(buffer_epsilon).buffer(-buffer_epsilon)

    if np.abs(poly_sh.area) < 1e-10:
        return None

    # Replace with convex hull if only small change in area.
    diff_poly_sh = poly_sh.convex_hull.difference(poly_sh)
    diff_ptc = diff_poly_sh.area / poly_sh.area
    if 0.0 < diff_ptc < convex_diff_pct:
        poly_sh = poly_sh.convex_hull

    if simplify_tol:
        poly_sh = poly_sh.simplify(simplify_tol, preserve_topology=False)

    if np.abs(poly_sh.area) < 1e-10:
        return None

    if not poly_sh.exterior.is_ccw:
        poly_sh = geom.polygon.orient(poly_sh, sign=1.0)

    return poly_sh


def poly_sh_rtree(poly_sh_arr):
    ridx = index.Index()
    for i, poly_sh in enumerate(poly_sh_arr):
        ridx.insert(i, poly_sh.bounds)
    return ridx


def almost_equals(poly1: np.ndarray, poly2: np.ndarray,
                  equal_tol: float = 2.0, scale: float = 1.0) -> bool:
    """Check if duplicate polygon."""

    poly1, poly2 = poly1.T, poly2.T
    equal_tol /= scale
    poly1_pts_num, poly2_pts_num = poly1.shape[0], poly2.shape[0]
    if poly1_pts_num != poly2_pts_num:
        return False

    pt1_idx, pt2_idx = None, None
    for i, pt1 in enumerate(poly1):
        for j, pt2 in enumerate(poly2):
            if np.allclose(pt1, pt2, atol=equal_tol):
                pt1_idx, pt2_idx = i, j
                break

    if pt1_idx is None:
        return False

    if pt1_idx != 0:
        poly1_idxs = list(range(poly1_pts_num))
        poly1_idxs = poly1_idxs[:pt1_idx] + poly1_idxs[pt1_idx:]
        poly1 = poly1[poly1_idxs]

    if pt2_idx != 0:
        poly2_idxs = list(range(poly2_pts_num))
        poly2_idxs = poly2_idxs[:pt2_idx] + poly2_idxs[pt2_idx:]
        poly2 = poly2[poly2_idxs]

    return np.allclose(poly1, poly2, atol=equal_tol)


def filter_parent_polys(polys_sh, scale=1):
    """Check if rtree contains polygon.

    Args:
        keep_equal_child: bool to indicate to replace parent with child
    """
    polys_sh = [p.buffer(0) for p in polys_sh]
    ridx = poly_sh_rtree(polys_sh)
    _polys_sh = []
    for poly_sh in polys_sh:
        hits = ridx.contains(poly_sh.bounds)
        hits = list(hits)

        # get rid of false positives
        hits = [h for h in hits if poly_sh.contains(
            polys_sh[h].buffer(-0.5 * scale))]

        if len(hits) <= 1:
            # child polygons, 1 indicates they contain themselves
            _polys_sh.append(poly_sh)

    return _polys_sh


def filter_duplicate_polys(ridx, polys_sh, equal_tol=1.0, scale=1.0):
    """Check for duplicate polygons."""

    equal_tol /= scale
    _polys_sh, _dups = [], set()
    num_polys = len(polys_sh)
    polys_sh_np = [to_poly_np(poly_sh) for poly_sh in polys_sh]

    for i, (poly_sh_np, poly_sh) in enumerate(zip(polys_sh_np, polys_sh)):

        if i in _dups:
            continue

        hits = ridx.intersection(poly_sh.bounds)
        dups_hits = [h for h in hits
                     if almost_equals(poly_sh_np, polys_sh_np[h], equal_tol)]

        # add poly to list
        _polys_sh.append(poly_sh)
        # add potential duplicates to set
        _dups |= set(dups_hits)

    return _polys_sh


def filter_by_size(poly_sh_arr, area_tol=None, side_tol=None, scale=1.0):

    def _is_side_fx(side_tol, scale):
        def _is_side_fx_helper(p):
            minx, miny, maxx, maxy = p.bounds
            width, height = (maxx - minx) * scale, (maxy - miny) * scale
            return width > side_tol and height > side_tol
        return _is_side_fx_helper

    if area_tol:
        poly_sh_arr = [p for p in poly_sh_arr
                       if (p.area * scale * scale) > area_tol]

    if side_tol:
        is_side_fx = _is_side_fx(side_tol, scale)
        poly_sh_arr = [p for p in poly_sh_arr if is_side_fx(p)]

    return poly_sh_arr


def gauss_mixture_1d(pts_1d: np.ndarray, n_components: float = 10,
                     init_var: float = 0.5, random_seed=None) -> GaussianMixture:
    """Get 1d gaussian mixture model from 1D points along x-axis.

        Args:
            pts_1d: (n_pts, 2) array of np.float64 type. Points must be projected onto x-axis.
            n_components: Number of clusters. Default 10.
            init_var: Variance to initialize variance of x-axis. Default is 0.5, which is
                wider then tolerance for points indicate.
            random_seed: Integer for random seed, default is None.
        Returns:
            GaussianMixture model of 1D clusters.
    """

    if (len(pts_1d.shape) != 2) or (pts_1d.shape[1] != 2) or (not pts_1d.dtype == np.float64):
        raise Exception('pts_1d must be a 2D float array of shape (n_pts, 2). '
                        'Got: {} shape of type {}.'.format(pts_1d.shape, pts_1d.dtype))

    if not np.allclose(pts_1d[:, 1], np.zeros_like(pts_1d.shape[0]), 1e-10):
        raise Exception(
            'pts_1d must be projected to x-axis, got non-zero values on y-axis.')

    # 2x average sum of squares from cluster to pt w/i tol
    precs = np.ones((n_components, 2)) * 1e-10
    precs[:, 0] = 1 / init_var  # 1d variance, wider works better

    gauss_model = GaussianMixture(
        n_components=n_components, covariance_type='diag',
        random_state=random_seed, warm_start=True,
        precisions_init=precs)

    return gauss_model.fit(pts_1d)


def contiguous_ones_idx(bit_arr: np.ndarray) -> np.ndarray:
    """Find indices of contiguous (repeated) ones in bit array.

    Given binary array (0s, 1s), returns (start, end) indices of
    repeated blocks of ones:

    bin_arr = np.array([1, 0, 0, 1, 0, 0, 1, 1])
    contiguous_ones_idx(bin_arr) -> [[0, 1] [3, 4] [6, 8]]
    """
    bit_arr = (bit_arr / bit_arr.max()).astype(np.bool)
    bit_arr = np.concatenate(([0], bit_arr, [0]))
    diff = np.abs(np.diff(bit_arr))
    zero_idx = np.where(diff == 1)[0].reshape(-1, 2)
    return zero_idx


def whitespace_pixel_dist(pixel_arr, viz=False, **kwargs):
    """Return distance whitespace as array from 1d pixel array."""
    pixel_arr = pixel_arr.astype(int)
    zz = np.ones((int(np.round(pixel_arr.max())) + 1)).astype(np.bool)
    zz[pixel_arr] = 0
    white_space_idx = contiguous_ones_idx(zz)
    white_space_idx.T

    if viz:
        _, a = plt.subplots()
        a.imshow(np.array([zz] * 3), **kwargs)

    # distance of white space. Less then 16 = close
    pix_gap = white_space_idx[:, 1] - white_space_idx[:, 0]
    return pix_gap


def estimate_1d_cluster_num(xproj: np.ndarray, yproj: np.ndarray, gap_tol: float,
                            viz: bool = False) -> np.ndarray:
    """Estimate x andy components from x and y projection and gap filtering.

    Args:
        xproj: A (N, 2) array of points with y axis-0.
        yproj: A (N, 2) array of points with x axis=0.
        gap_tol: float

    Returns:
        Two numbers for x and y cluster component numbers.

    """
    # Calc distance of white spaces
    x_white_dist_arr = whitespace_pixel_dist(xproj[:, 0], viz=viz)
    y_white_dist_arr = whitespace_pixel_dist(yproj[:, 1], viz=viz)

    # Cluster components by filtering out small gaps
    x_comp = np.where(x_white_dist_arr > gap_tol)[0].size
    y_comp = np.where(y_white_dist_arr > gap_tol)[0].size

    return x_comp, y_comp


def xy_projection(poly_sh_arr):
    """Produces two 2d arrays of x, y points, where one axis is 0."""
    # Get model points
    _xproj = np.concatenate([to_poly_np(poly_sh)[0]
                            for poly_sh in poly_sh_arr])
    _yproj = np.concatenate([to_poly_np(poly_sh)[1]
                            for poly_sh in poly_sh_arr])

    # Create 2D x, y projection matrices
    n_pts = _xproj.shape[0]
    xproj, yproj = np.zeros((n_pts, 2)), np.zeros((n_pts, 2))
    xproj[:, 0] = _xproj
    yproj[:, 1] = _yproj

    return xproj, yproj


def move_poly_sh_arr(poly_sh_arr, xdim=None, ydim=None):
    """Moves multiple poly_sh by xdim, ydim.

    If None for xdim or ydim, that axis will be moved to origin.
    """

    if xdim is None or ydim is None:
        xproj, yproj = xy_projection(poly_sh_arr)
        xdim = -xproj[:, 0].min() if xdim is None else xdim
        ydim = -yproj[:, 1].min() if ydim is None else ydim

    return [affinity.translate(p, xdim, ydim) for p in poly_sh_arr]


def get_cluster_coord_fx(x_gauss_mod, y_gauss_mod):
    """Get fx coordinates of closest cluster for polygon pts from gaussian mixture model."""
    x_means = x_gauss_mod.means_[:, 0].round(2)
    y_means = y_gauss_mod.means_[:, 0].round(2)

    def _get_cluster_coord_fx(polyc):
        polyc_xproj = np.array([polyc[:, 0], np.zeros_like(polyc[:, 0])]).T
        polyc_yproj = np.array([polyc[:, 1], np.zeros_like(polyc[:, 1])]).T
        x_clusters = x_gauss_mod.predict(polyc_xproj)
        y_clusters = y_gauss_mod.predict(polyc_yproj)

        predicted_pts = np.array([x_means[x_clusters], y_means[y_clusters]])
        return predicted_pts

    return _get_cluster_coord_fx


def extract_floorplan_ids(data_num):
    """Safely extract root model directories for polygon extraction."""

    # Load all model directories
    floorplan_id_arr = os.listdir(DEEPRAD_DATA_DIR)
    n_ids = len(floorplan_id_arr)

    if n_ids == 0:
        raise Exception('No image files. Add images and try again.')

    if data_num > n_ids:
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
            DEEPRAD_DATA_DIR, targ_id_dirs[i], 'data')
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
        src_img_arr[idx] = cv2.imread(targ_src_fpath, cv2.COLOR_BGR2RGB)
        label_img_arr[idx] = cv2.imread(targ_label_fpath, cv2.IMREAD_GRAYSCALE)
        targ_id_dir_arr[idx] = targ_id_dir
        null_lst.append(targ_id_dirs[i] + '\n')

    # Write to null list
    null_fpath = os.path.join(DEEPRAD_DATA_DIR, '_null.txt')
    with open(null_fpath, 'w') as fp:
        #[fp.writeline(null_) for null_ in null_lst]
        fp.writelines(null_lst)
    return hdict_arr, src_img_arr, label_img_arr, targ_id_dir_arr


def main(data_num):
    """Generate data_num amount of floorplan polygon vectors."""

    data_num, targ_ids = extract_floorplan_ids(data_num)
    hdicts, _, labels, targ_id_dirs = load_floorplan_data(targ_ids, data_num)

    print('\nWriting {} polygon json files in {}.\n'.format(
          data_num, DEEPRAD_DATA_DIR))

    for ii, (hdict, _rooms, targ_id_dir) in enumerate(zip(hdicts, labels, targ_id_dirs)):

        model_id = hdict['id']

        # -------------------------------------------------------------
        # Extract/remove image data
        # -------------------------------------------------------------
        scale = hdict['scale']
        doors = hdict['door_vecs']

        # Remove balconies, railings, and walls
        _rooms = np.where(_rooms == 1, 0, _rooms)
        _rooms = np.where(_rooms == 8, 0, _rooms)
        _rooms = np.where(_rooms == 2, 0, _rooms)

        # -------------------------------------------------------------
        # Get door_vec data for scaling
        # -------------------------------------------------------------
        door_lens = []
        for door in doors:
            pts, _ = door[0], door[1]
            pts = np.array(pts) * scale
            door_vec = pts[1] - pts[0]
            door_len = np.sum(door_vec ** 2) ** 0.5
            door_lens.append(door_len)

        door_lens = np.array(door_lens)
        meter_scale = 0.9 / door_lens.mean()
        image_scale = 1 / meter_scale

        # Gut check the scale
        _dim = np.where(np.sum(_rooms, axis=0) > 1, 0, 1)
        _dim = contiguous_ones_idx(_dim)
        diff = _dim[:, 1] - _dim[:, 0]
        diff = np.diff(_dim, n=1, axis=1).ravel()
        _dim = _rooms.shape[1] - np.sum(diff)

        # -------------------------------------------------------------
        # Take first order derivative for thresholding.
        # -------------------------------------------------------------
        room = _rooms.astype(np.float64)
        dx = np.abs(np.diff(room, axis=0, prepend=0.0))
        dy = np.abs(np.diff(room, axis=1, prepend=0.0))
        diff = dx + dy
        diff = np.where(diff > 0, 255, 0)
        threshed_rooms = diff.astype(np.uint8)

        # -------------------------------------------------------------
        # Get contour w/ opencv.
        # -------------------------------------------------------------
        _contour_arr = contour_arr_from_bit_image(
            threshed_rooms, threshold=True)[:-1]
        _poly_np_arr = [contour_to_poly_np(contour)
                        for contour in _contour_arr]
        _poly_sh_arr = [to_poly_sh(poly_np) for poly_np in _poly_np_arr]

        beps, stol, dpct = 5, 0.25 * image_scale, 0.05
        def _clean_poly_sh(p): return clean_poly_sh(p, beps, stol, dpct)
        _poly_sh_arr = [_clean_poly_sh(_poly_sh)
                        for _poly_sh in _poly_sh_arr]
        _poly_sh_arr = [p for p in _poly_sh_arr if p is not None]

        # -------------------------------------------------------------
        # Scale and shift
        # -------------------------------------------------------------
        # TODO: do this at the beginning after doors.
        poly_sh_arr = [copy(p) for p in _poly_sh_arr]
        unscaled_poly_sh_arr = [copy(p) for p in _poly_sh_arr]
        poly_sh_arr = move_poly_sh_arr(poly_sh_arr)
        poly_sh_arr = [affinity.scale(
            p, meter_scale, meter_scale, origin=(0, 0)) for p in poly_sh_arr]

        # -------------------------------------------------------------
        # Filter polygon checks.
        # -------------------------------------------------------------

        n_poly_sh = len(poly_sh_arr)
        ridx = poly_sh_rtree(poly_sh_arr)
        # print('num @', n_poly_sh)

        # TODO: set scale tol
        gap_tol = 1.0
        poly_sh_arr = filter_duplicate_polys(ridx, poly_sh_arr, gap_tol)
        # _dup_diff = n_poly_sh - len(poly_sh_arr)
        # print('Removed:', _dup_diff, 'duplicates; num @', len(poly_sh_arr))

        area_tol, side_tol = 2, 1
        poly_sh_arr = filter_by_size(
            poly_sh_arr, area_tol, side_tol, 1)
        # _sml_diff = n_poly_sh - _dup_diff - len(poly_sh_arr)
        # print('Removed:', _sml_diff, 'smalls; num @', len(poly_sh_arr))

        poly_sh_arr = filter_parent_polys(poly_sh_arr, 1)
        # _par_diff = n_poly_sh - _sml_diff - len(poly_sh_arr)
        # print('Removed:', _par_diff, 'parents; num @', len(poly_sh_arr))
        # print('num: ', len(poly_sh_arr))

        # -------------------------------------------------------------
        # Compute projections.
        # -------------------------------------------------------------

        xproj_, yproj_ = xy_projection(unscaled_poly_sh_arr)
        xproj, yproj = xy_projection(poly_sh_arr)

        # Calculate cluster number for x/y axis
        gap_tol = 0.5 * image_scale
        x_ncomps, y_ncomps = estimate_1d_cluster_num(
            xproj_, yproj_, gap_tol, False)
        # print('x comp: {}, y comp: {}'.format(x_ncomps, y_ncomps))

        _poly_sh_arr = poly_sh_arr

        # -------------------------------------------------------------
        # Compute gaussian mixture models
        # -------------------------------------------------------------

        random_seed = 111
        poly_sh_arr = [copy(p) for p in _poly_sh_arr]

        # swap axis on yproj to be on xaxis
        _FLIP_ORTHO2D_MTX = np.array([[0, 1], [1, 0]])
        yproj_swap = yproj @ _FLIP_ORTHO2D_MTX

        # tol = 25.0  # door width
        # tar_var = (tol / 2) ** 2
        init_var = 2.0
        x_gauss_mod = gauss_mixture_1d(
            xproj, x_ncomps, init_var, random_seed)
        y_gauss_mod = gauss_mixture_1d(
            yproj_swap, y_ncomps, init_var, random_seed)

        # -------------------------------------------------------------
        # Predict cluster and snap to orthogrid
        # -------------------------------------------------------------
        _cluster_fx = get_cluster_coord_fx(x_gauss_mod, y_gauss_mod)
        buff_eps, simple_tol, diff_pct = 1, 0.5, 0.05

        poly_sh_arr_ = []
        for poly_sh in poly_sh_arr:
            poly_np = _cluster_fx(to_poly_np(poly_sh).T)
            polyc_sh = to_poly_sh(poly_np)
            polyc_sh = clean_poly_sh(polyc_sh, buff_eps, simple_tol, diff_pct)
            if polyc_sh is None or polyc_sh.area < 1e-10 or polyc_sh.exterior is None:
                continue
            poly_sh_arr_.append(polyc_sh)

        poly_sh_arr = poly_sh_arr_

        # -------------------------------------------------------------
        # Dump data.
        # -------------------------------------------------------------
        def write_json(polygon_dict, dest_fpath):
            """Writes dict to json."""
            with open(dest_fpath, 'w') as fp:
                json.dump(polygon_dict, fp, indent=4)

        # Dump polygons as numpy arrays
        polygon_json_fpath = os.path.join(targ_id_dir, 'polygon.json')
        polygon_dict = {'polygons': [to_poly_np(poly_sh).tolist()
                                     for poly_sh in poly_sh_arr]}
        write_json(polygon_dict, polygon_json_fpath)

        # -------------------------------------------------------------
        # Finish
        # -------------------------------------------------------------
        print('{}/{}: wrote {} polygon to {}'.format(
              ii + 1, data_num, targ_ids[ii], polygon_json_fpath))


if __name__ == "__main__":

    # Set defaults
    data_num = 1

    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        if '--data_num' in argv:
            data_num = int(argv[1])

    main(data_num)
