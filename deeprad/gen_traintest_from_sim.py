import os
import sys
import json

git_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
pths = [os.path.join(git_dir, 'pincam')]

for pth in pths:
    if pth not in sys.path: sys.path.insert(0, pth)

from astrobot.util import *
from astrobot.geomdataframe import GeomDataFrame
from astrobot.r import R
from astrobot import bem_util, geom_util, mtx_util
#from astrobot.polymesh import PolyMesh

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
#import itertools

import ladybug_geometry.geometry3d as geom3
import ladybug_geometry.geometry2d as geom2
from ladybug_geometry_polyskel import polyskel

# Simulation
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter

import honeybee.dictutil as hb_dict_util
from ladybug.epw import EPW

# For camera
from shapely.geometry import Polygon
from pincam import Pincam

from ladybug.sql import SQLiteResult
from honeybee_energy.result import match


def poly_in_cam_view(camP, poly_np_arr, in_view=True):
    """Return boolean mask of in camera view from poly_np_arr.

    Does not account for depth obstruction, purely a view factor calc.
    Given poly_np_arr and and a pincam projection matrix (cam.P).
    """

    view_set = set()
    mask = np.arange(len(poly_np_arr))

    for mask_idx, poly_np in zip(mask, poly_np_arr):
        view_factor = Pincam.view_factor(camP, poly_np)
        if in_view:
            if view_factor < 0.0: #or r.srf.loc[mask_idx, 'type']  == 'Floor':
                view_set.add(mask_idx)
        else:
            if view_factor > 0.0:
                view_set.add(mask_idx)

    # Mask in views
    return [(mask_idx in view_set) for mask_idx in mask]

def _cam_color(poly_sh_arr, color, a, **kwargs):
    """Color from string or array of strings."""
    a = GeomDataFrame({'geometry': poly_sh_arr}).plot(
        color=color, edgecolor='black', ax=a, **kwargs)
    a.axis('equal'); a.axis('off')
    return a

def to_poly_np(poly_sh, lb=True):
    """to poly numpy array of vertices."""
    if lb:
        return np.array([v.to_array() for v in poly_sh.vertices])
    else:
        return np.array([*poly_sh.exterior.xy]).T

def _project(P, pmtx, ortho=True, res=2):
    """Wrapper for pincam projection."""
    xformed_ptmtx, depths = Pincam.project(P, pmtx, ortho=ortho)
    #depths, db = cam.depth_buffer(ptmtx, _depths, res=res)
    xformed_ptmtx = [xformed_ptmtx[d] for d in depths]
    return [Pincam.to_poly_sh(poly_np) for poly_np in xformed_ptmtx], depths

def _int_geom_df(srf_df):
    """Query df of interior surfaces."""
    return srf_df.query('bc == "Surface" or bc == "Adiabatic"').index

def _ext_geom_df(srf_df):
    """Query df of exterior surfaces."""
    return srf_df.query('bc != "Surface"').index

def _cam_cmap(poly_sh_arr, cmap_arr, f, a, **kwargs):
    """Cmap from values."""
    # normalize color
    cmap = 'RdYlBu_r'
    srf_df = GeomDataFrame({'geometry': poly_sh_arr, 'var': cmap_arr})
    a = srf_df.plot(column='var', edgecolor='black',
                    cmap=cmap, legend=False, ax=a, **kwargs)
    a.axis('equal'); a.axis('off')
    return a

def deg2rad(d):
    return d / 180. * np.pi

def reorder_to_srf_geom_arr(r_srf_geom_arr, srf_mod_idx, srf_sim_data, div_by_area=True):
    """Match data coll to r_srf order.

    Returns filtered srf_mod_idx and data that you can use like this:

    ```
    r_srf_geom_arr = r.srf.loc[srf_mod_idx, 'srf_geom']
    _fil_mod_idx, _fil_data = reorder_to_srf_geom_arr(r_srf_geom_arr, srf_mod_idx, srf_sim_data)
    r.srf.loc[_fil_mod_idx, 'srf_heat_transfer'] = _fil_data
    ```
    """
    # Match surfaces
    mod_idx = srf_mod_idx
    matched = match.match_faces_to_data(srf_sim_data, r_srf_geom_arr)
    matched_ids = {s.identifier.upper: d for s, d in matched}
    rematched_data = np.empty(len(srf_mod_idx))
    rematched_data[:] = np.NaN
    srf_area = 1.0
    for i, srf in enumerate(r_srf_geom_arr):
        srf_id = srf.identifier.upper

        if div_by_area:
            srf_area = srf.area

        if srf_id not in matched_ids:
            continue

        data = matched_ids[srf_id]
        val = (data.total * 2) / srf_area
        #print(data)
        #print(data.values, srf_area)
        #print('--')
        rematched_data[i] = val

    # update mod_idx to remove internals by filter out nans
    #print(len(mod_idx))
    _not_nan_idx = np.where(~np.isnan(rematched_data))
    filtered_mod_idx = srf_mod_idx[_not_nan_idx]
    filtered_data = rematched_data[_not_nan_idx]
    return filtered_mod_idx, filtered_data


def flattened_bems(model_fpaths):
    """Get flattened list of HB models from dir"""
    models, model_sims = [], []
    for i, model_id_fpath in enumerate(model_fpaths):
        build_ids = os.listdir(model_id_fpath)
        for j, build_id in enumerate(build_ids):
            build_id_fpath = os.path.join(model_id_fpath, build_id, 'OpenStudio')
            build_json_fpath = os.path.join(build_id_fpath, build_id + '.hbjson')
            sim_fpath = os.path.join(build_id_fpath, 'run', 'eplusout.sql')
            # check if simulated
            #if not os.path.isfile(sim_fpath):
            #    print('missing {} at {}'.format(i, sim_fpath))

            # Add model
            with open(build_json_fpath, 'r') as json_file:
                data = json.load(json_file)
            models.append(hb_dict_util.dict_to_object(data, False))
            model_sims.append(sim_fpath)
    return models, model_sims


def main(num):
    """Make facade traintest data from hbjsons."""

    RDD_SRF_DICT = {
        'srf_win_heat_loss': 'Surface Window Heat Loss Energy',
        'srf_win_heat_gain': 'Surface Heat Window Gain Energy',
        'srf_heat_transfer': 'Surface Average Face Conduction Heat Transfer Energy',
        'srf_win_sol': 'Surface Window Transmitted Solar Radiation Energy',
        'srf_ext_sol': 'Surface Outside Face Incident Solar Radiation Rate per Area',
        'srf_int_sol': 'Surface Inside Face Solar Radiation Heat Gain Rate',
        'srf_cos': 'Surface Outside Face Beam Solar Incident Angle Cosine Value'
        }

    rdd_srf_var = RDD_SRF_DICT['srf_int_sol']
    div_by_area = False if 'per Area' in rdd_srf_var else True
    vmin = 0
    vmax = 50 if 'Inside' in rdd_srf_var else 300
    print('Plot {} with div_by_area {}.'.format(rdd_srf_var, div_by_area))
    print('Outside should be 0-300, Inside should be 0-50. '
        'Values clipped between {} - {}'.format(vmin, vmax))
    show_legend = False
    NN2 =True

    # ---------------------------------------------------------------------------------
    # Get al HB models and simfpaths
    # ---------------------------------------------------------------------------------

    deeprad_hbjsons_dir = \
        os.path.join(os.getcwd(), '../../..', 'master/git', 'deeprad/data/hbjsons/')
    deeprad_hbjsons_dir = os.path.abspath(deeprad_hbjsons_dir)
    traintest_dir = os.path.join(deeprad_hbjsons_dir, '..', 'traintest3')

    _model_fpaths = sorted(os.listdir(deeprad_hbjsons_dir),
        key=lambda s: int(s.split('_')[0]))
    _model_fpaths = [os.path.join(deeprad_hbjsons_dir, mf) for mf in _model_fpaths]

    _model_fpaths = _model_fpaths
    if num:
        _model_fpaths = _model_fpaths[:num]

    models, model_sims = flattened_bems(_model_fpaths)
    model_num = len(models)

    # ---------------------------------------------------------------------------------
    # Init r.mod
    # ---------------------------------------------------------------------------------

    # make R
    mod_df = GeomDataFrame(
        {'model_id': [m.identifier for m in models],
         'model': models,
         'sim_fpath': model_sims})

    r = R(mod_df)
    model_num = len(r.mod)
    r.mod['null'] = np.ones(model_num)

    print('# of models', model_num)

    # ---------------------------------------------------------------------------------
    # Init r.srf
    # ---------------------------------------------------------------------------------

    # Add srfs
    faces, face_mod_idxs = [], []
    for i, model in enumerate(mod_df['model']):
        faces.extend(model.faces)
        face_mod_idxs.extend([i] * len(model.faces))

    # TODO: Mtx > mtx_util, Mtx_geom > mtx_geom_util, mtx_geom, move mtx_utli to Mtx?
    r.srf = GeomDataFrame({'mod_idx': face_mod_idxs, 'srf_geom': faces})
    r.srf['type'] = [srf.type.name for srf in r.srf['srf_geom']]
    r.srf['bc'] = [srf.boundary_condition.name for srf in r.srf['srf_geom']]
    srf_num = len(faces)


    # ---------------------------------------------------------------------------------
    # Add sim data to r.srf
    # ---------------------------------------------------------------------------------

    # Extract srf sim data
    r.srf['srf_heat_transfer'] = np.empty(srf_num)
    r.srf['srf_heat_transfer'] = np.NaN

    # Update surfaces for each model
    for i, sim_fpath in enumerate(r.mod['sim_fpath']):

        # Get exterior faces
        srf_mod_idx = r.srf.query('mod_idx == {}'.format(i)).index
        if not os.path.isfile(sim_fpath):
            continue

        sql = SQLiteResult(sim_fpath)
        srf_sim_data = sql.data_collections_by_output_name(rdd_srf_var)

        r_srf_geom_arr = r.srf.loc[srf_mod_idx, 'srf_geom']
        _fil_mod_idx, _fil_data = reorder_to_srf_geom_arr(
            r_srf_geom_arr, srf_mod_idx, srf_sim_data, div_by_area=div_by_area)
        r.srf.loc[_fil_mod_idx, 'srf_heat_transfer'] = _fil_data


    # ---------------------------------------------------------------------------------
    # Plot facade
    # ---------------------------------------------------------------------------------
    # Define constant camera properteis
    FOCAL_LEN = 45.0
    PITCH = deg2rad(15)  # Z
    CAM_POSN = np.array([0, -35, 2.5])  # camera placed at 2nd floor
    angle_iter = 45
    CAM_ANGLES = np.arange(0, 180 + angle_iter, angle_iter)
    PLT_NUM, PLT_HT = len(CAM_ANGLES), 4
    ORTHO = True

    def _cam_cmap(poly_sh_arr, cmap_arr, a, **kwargs):
        """Cmap from values."""
        # normalize color
        cmap = 'RdYlBu_r'
        srf_df = GeomDataFrame({'geometry': poly_sh_arr, 'var': cmap_arr})
        a = srf_df.plot(column='var', edgecolor='black',
                        cmap=cmap, ax=a, **kwargs)
        a.axis('equal'); a.axis('off')
        return a

    img_in_dir = os.path.join(traintest_dir, 'in_data')
    img_out_dir = os.path.join(traintest_dir, 'out_data')

    if NN2:
        img_in_dir = os.path.join(traintest_dir, 'in_data2')
        img_out_dir = os.path.join(traintest_dir, 'out_data2')

    for mod_idx in r.mod.index:
        try:
            sim_fpath = r.mod.loc[mod_idx, 'sim_fpath']
            hb_idx = sim_fpath.split('_hb')[0].split('_')[-1].split('/')[-1]
            img_out_fpath = os.path.join(img_out_dir, '{}_{}_hb_solrad_out.jpg'.format(mod_idx, hb_idx))
            img_in_fpath1 = os.path.join(img_in_dir, '{}_{}_hb_facetype_in.jpg'.format(mod_idx, hb_idx))
            img_in_fpath2 = os.path.join(img_in_dir, '{}_{}_hb_mask_in.jpg'.format(mod_idx, hb_idx))

            # ---------------------------------------------------------------------------------
            # Out
            # ---------------------------------------------------------------------------------

            # # TODO: delete, this is better solved through a monte-carlo simulation
            # # TODO: consider implement based on floor normal
            # # (see_top = plot last else plot first in story). I like monte-carlo better.
            # def _order_floors_last(srf_arr, type_arr, df=True):
            #     """Rearranges floors last in depth list so they are plotted in front of walls."""
            #     # TODO: need to move floors/roofs to appropriate story data.
            #     # TODO: but what about non-facing data..

            #     idx_arr = srf_arr.index \
            #         if isinstance(srf_arr, np.ndarray) else np.arange(len(srf_arr))
            #     is_floor = ['Floor' in st for st in type_arr]
            #     reorder_idx = np.concatenate([idx_arr[~is_floor], idx_arr[~is_floor]])
            #     return srf_arr[reorder_idx]

            if not NN2:
                f, a = plt.subplots(
                    1, PLT_NUM, figsize=(2 * PLT_HT * PLT_NUM, PLT_HT), sharey=True)
                plt.setp(a, yticks=np.arange(10, -10, 2))
                f.tight_layout()

                for i, cam_angle in enumerate(CAM_ANGLES):
                    cam = Pincam(CAM_POSN, deg2rad(cam_angle), PITCH, FOCAL_LEN)
                    #mask = _ext_geom_df(r.srf.query('mod_idx == {}'.format(mod_idx)))
                    mask = r.srf.query('mod_idx == {}'.format(mod_idx)).index
                    poly_np_arr = [to_poly_np(poly_lb, True) for poly_lb in r.srf.loc[mask, 'srf_geom']]
                    poly_sh_arr, depths = _project(cam.P, poly_np_arr, ortho=ORTHO, res=1)

                    #_srf_arr = _order_floors_last(*r.srf.loc[mask, ['srf_geom', 'type']].T.values)
                    #mask = _srf_arr.index

                    srf_heat_arr = r.srf.loc[mask, 'srf_heat_transfer'].values
                    srf_heat_arr = np.array([srf_heat_arr[d] for d in depths])
                    if not i:
                        #print(r.mod.loc[mod_idx, 'sim_fpath'])
                        print(srf_heat_arr[~np.isnan(srf_heat_arr)].min(), srf_heat_arr[~np.isnan(srf_heat_arr)].max())


                    srf_heat_arr[np.where(np.isnan(srf_heat_arr))] = 0
                    #color_arr = 'lightblue' #[ for ii in srf_df['srf_heat_transfer']]
                    #_cam_color(poly_sh_arr, color_arr, a[i], linewidth=4)
                    a[i] = _cam_cmap(poly_sh_arr, srf_heat_arr, a[i], linewidth=4,
                        vmin=vmin, vmax=vmax, legend=show_legend)

                f.savefig(img_out_fpath)
                f.clf()
                plt.close('all')
                print('Saved out img: {}'.format(img_out_fpath))

            else:
                f, a = plt.subplots(
                    1, PLT_NUM, figsize=(2 * PLT_HT * PLT_NUM, PLT_HT), sharey=True)
                plt.setp(a, yticks=np.arange(10, -10, 2))
                f.tight_layout()

                for i, cam_angle in enumerate(CAM_ANGLES):
                    cam = Pincam(CAM_POSN, deg2rad(cam_angle), PITCH, FOCAL_LEN)
                    mask = r.srf.query('mod_idx == {}'.format(mod_idx)).index

                    in_view = set()
                    for mask_idx, srf_geom in zip(mask, r.srf.loc[mask, 'srf_geom']):
                        verts = [v.to_array() for v in srf_geom.vertices]
                        view_factor = Pincam.view_factor(cam.P, verts)
                        if view_factor < 0.0 or r.srf.loc[mask_idx, 'type']  == 'Floor':
                            in_view.add(mask_idx)

                    mask = [mask_idx for mask_idx in mask if mask_idx in in_view]
                    poly_np_arr = [to_poly_np(poly_lb, True) for poly_lb in r.srf.loc[mask, 'srf_geom']]
                    poly_sh_arr, depths = _project(cam.P, poly_np_arr, ortho=ORTHO, res=1)
                    srf_heat_arr = r.srf.loc[mask, 'srf_heat_transfer'].values
                    srf_heat_arr = np.array([srf_heat_arr[d] for d in depths])

                    if not i:
                        #print(r.mod.loc[mod_idx, 'sim_fpath'])
                        minv, maxv = srf_heat_arr[~np.isnan(srf_heat_arr)].min(), srf_heat_arr[~np.isnan(srf_heat_arr)].max()
                        print('min: {}, max: {}'.format(np.round(minv, 2), np.round(maxv, 2)))

                    srf_heat_arr[np.where(np.isnan(srf_heat_arr))] = 0

                    #color_arr = 'lightblue' #[ for ii in srf_df['srf_heat_transfer']]
                    #_cam_color(poly_sh_arr, color_arr, a[i], linewidth=4)
                    a[i] = _cam_cmap(poly_sh_arr, srf_heat_arr, a[i], linewidth=4,
                        vmin=vmin, vmax=vmax, legend=show_legend)

                plt.savefig(img_out_fpath)
                plt.clf()
                plt.close('all')
                print('Saved out img: {}'.format(img_out_fpath))

            # plt.close('all')
            # # ---------------------------------------------------------------------------------
            # # In
            # # ---------------------------------------------------------------------------------
            if not NN2:
                f, a = plt.subplots(1, PLT_NUM, figsize=(2 * PLT_HT * PLT_NUM, PLT_HT), sharey=True)
                plt.setp(a, yticks=np.arange(10, -10, 2))
                f.tight_layout()
                for i, cam_angle in enumerate(CAM_ANGLES):
                    cam = Pincam(CAM_POSN, deg2rad(cam_angle), PITCH, FOCAL_LEN)
                    mask = r.srf.query('mod_idx == {}'.format(mod_idx)).index
                    poly_np_arr = [to_poly_np(poly_lb, True) for poly_lb in r.srf.loc[mask, 'srf_geom']]
                    poly_sh_arr, depths = _project(cam.P, poly_np_arr, ortho=ORTHO, res=1)
                    bcs = r.srf.loc[mask, 'bc'].values
                    bcs = np.array([bcs[d] for d in depths])

                    color_arr = [0] * len(bcs)
                    for k, _bc in enumerate(bcs):
                        if _bc == 'Adiabatic':
                            color_arr[k] = 'grey'
                        else:
                            color_arr[k] = 'lightblue'

                    _cam_color(poly_sh_arr, color_arr, a[i], linewidth=4)

                f.savefig(img_in_fpath1)
                f.savefig(img_in_fpath2)
                plt.clf()
                plt.close('all')
                print('Saved in img: {}'.format(img_in_fpath1))
                print('Saved in img: {}'.format(img_in_fpath2))
            else:
                srf_f, srf_a = plt.subplots(1, PLT_NUM, figsize=(2 * PLT_HT * PLT_NUM, PLT_HT), sharey=True)
                plt.setp(srf_a, yticks=np.arange(10, -10, 2))
                win_f, win_a = plt.subplots(1, PLT_NUM, figsize=(2 * PLT_HT * PLT_NUM, PLT_HT), sharey=True)
                plt.setp(win_a, yticks=np.arange(10, -10, 2))
                srf_f.tight_layout(); win_f.tight_layout()

                for i, cam_angle in enumerate(CAM_ANGLES):
                    # Get surfaces
                    mask = r.srf.query('mod_idx == {}'.format(mod_idx)).index
                    srf_geoms = r.srf.loc[mask, 'srf_geom']
                    _win_geoms = [srf_geom.apertures for srf_geom in srf_geoms]
                    _win_geoms = [j for i in _win_geoms for j in i]
                    win_geoms = [win_geom.duplicate().geometry.move(win_geom.normal.duplicate() * 0.001)
                                for win_geom in _win_geoms]

                    srf_np_arr = [to_poly_np(poly_lb, True) for poly_lb in srf_geoms]
                    win_np_arr = [to_poly_np(poly_lb, True) for poly_lb in win_geoms]

                    # Camera
                    cam = Pincam(CAM_POSN, deg2rad(cam_angle), PITCH, FOCAL_LEN)
                    # filter srfs by in view
                    srf_np_arr = [to_poly_np(poly_sh) for poly_sh in r.srf.loc[mask, 'srf_geom']]
                    mask_bool = poly_in_cam_view(cam.P, srf_np_arr, in_view=True)
                    view_srf_np_arr = [srf_np for m, srf_np in zip(mask_bool, srf_np_arr) if m]
                    view_mask = [_m for b, _m in zip(mask_bool, mask) if b]

                    # Project
                    view_srf_sh_arr, view_srf_depths = _project(cam.P, view_srf_np_arr, ortho=ORTHO, res=1)
                    win_sh_arr, win_depths = _project(cam.P, win_np_arr, ortho=ORTHO, res=1)
                    srf_sh_arr, srf_depths = _project(cam.P, srf_np_arr, ortho=ORTHO, res=1)
                    all_sh_arr, all_depths = _project(cam.P, srf_np_arr + win_np_arr, ortho=ORTHO, res=1)

                    # label depths by type
                    win_start_idx = len(srf_np_arr)
                    type_str = ['win' if d >= win_start_idx else 'srf' for d in all_depths]
                    type_dict = dict(zip(all_depths, type_str))

                    # colors for view srfs
                    vbcs = r.srf.loc[view_mask, 'bc'].values
                    vbcs = np.array([vbcs[d] for d in view_srf_depths])
                    view_srf_color_arr = [0] * len(view_srf_sh_arr)
                    for k, _vbc in enumerate(vbcs):
                        view_srf_color_arr[k] = 'grey' if _vbc == 'Adiabatic' else 'lightblue'

                    # colors for srfs
                    abcs = r.srf.loc[mask, 'bc'].values
                    abcs = np.array([abcs[d] for d in srf_depths])
                    srf_color_arr = [0] * len(srf_sh_arr)
                    for k, _abc in enumerate(abcs):
                        srf_color_arr[k] = 'grey' if _abc == 'Adiabatic' else 'lightblue'

                    # colors for wins
                    bcs = r.srf.loc[mask, 'bc'].values
                    bcs = np.array([bcs[d] for d in srf_depths])
                    srf_i = 0
                    all_color_arr = [0] * len(all_sh_arr)
                    for k, (all_sh, all_depth) in enumerate(zip(all_sh_arr, all_depths)):
                        if type_dict[all_depth] == 'win':
                            all_color_arr[k] = 'red'
                        else:
                            all_color_arr[k] = 'white' if bcs[srf_i] == 'Adiabatic' else 'white'
                            srf_i += 1

                    # Ghost the building
                    srf_a[i] = _cam_color(view_srf_sh_arr, view_srf_color_arr, srf_a[i], linewidth=4)
                    srf_a[i] = _cam_color(srf_sh_arr, srf_color_arr, a=srf_a[i], linewidth=4, alpha=0.5)

                    # second channel
                    _cam_color(all_sh_arr, all_color_arr, win_a[i], linewidth=0)
                    del _win_geoms; del win_geoms

                srf_f.savefig(img_in_fpath1)
                win_f.savefig(img_in_fpath2)
                plt.clf()
                plt.close('all')
                print('Saved in img: {}'.format(img_in_fpath1))

        except Exception as e:
            print('\nFail at {} {}\n'.format(mod_idx, e))

if __name__ == "__main__":

    arg_dict = {
        'num': False
        }

    if len(sys.argv) > 1:
        argv = sys.argv[1:]

        for arg in argv:
            if '--' not in arg: continue
            assert arg.split('--')[-1] in arg_dict, \
                '{} not an arg: {}'.format(arg, arg_dict.keys())

        if '--num' in argv:
            i = argv.index('--num')
            arg_dict['num'] = int(argv[i + 1])

    print('\nPress Enter to confirm user arg:')
    [print('{}: {}'.format(k, v)) for k, v in arg_dict.items()]
    input('...')


    num = arg_dict['num']
    main(num)