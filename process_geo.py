
from honeybee.face import Face
from ladybug_rhino.fromgeometry import from_face3d, from_polyface3d, from_point3d, from_vector3d
import ladybug_rhino.fromgeometry as rfg
import ladybug_rhino.togeometry as rtg
import ladybug_geometry as lg
from ladybug_geometry import geometry3d as geom3d
from ladybug_geometry import geometry2d as geom2d
from ladybug_geometry.geometry2d.pointvector import Vector2D
from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D
from ladybug_geometry.geometry3d import Face3D
from ladybug_geometry.geometry3d import face


import math


def get_normals(solid):
    lb_face = rtg.to_face3d(solid)

    faces = []
    pts = []
    vecs = []
    
    north_vec = Vector2D(0,1)
    dir = ['North', 'East', 'South', 'West','North' ]

    angles = [[0,45],[45,135], [135,225], [225,315], [315,360]]
    orientations = []
    c = 0
    for f in lb_face:
        fvec = from_point3d(f.normal)
        z = fvec[2]
#        print(z)
        if z != 1.0 and z != -1.0:
            c = c+1
            
            face = rfg.from_face3d(f)
            pt = from_vector3d(f.center)
            axis = Vector3D(0,0,1)
            pts.append(pt)
            ang_diff = north_vec.angle_clockwise(Vector2D(fvec[0],fvec[1]))
            orient = math.degrees(ang_diff)
            f_rot = f.rotate(axis, ang_diff, f.center)
            face = rfg.from_face3d(f_rot)
            faces.append(face)
            
            for i, ang in enumerate(angles):
             
                if orient> ang[0] and orient<=ang[1]:
                    orientations.append(dir[i])
                    break

#                else:
#                    orientations.append(dir[0])
#                    break
           
#            vec = vct.rotate_xy(axis, ang_diff, f.center)
            vct = from_point3d(f_rot.normal)
#            vecs.append(fvec)
            vecs.append(vct)

    dir_index = [dir.index(i) for i in orientations]
    
    return faces, pts, vecs, orientations, dir_index

def align(faces, cntrs):
    
    face_list = []
    face_list.append(faces[0])
    for i, f in enumerate (faces):
        if i >0:
            lf = rtg.to_face3d(f)[0]
            lfo = rtg.to_face3d(f)[0]
            if i ==1:
                lfp = rtg.to_face3d(faces[i-1])[0]
            else:
                lfp = rtg.to_face3d(face_list[i-1])[0]
            
            _x = lfp.min[0]  - lf.min[0]
            _y = lfp.min[1]  - lf.min[1]
            _z = 0
            m_vec = Vector3D(_x, _y, 0)

            lf1 =lf.move(m_vec)
            
            mnp = lf1.min
            mxp = lf1.max
            
            
            pv_seg = lfp.get_top_bottom_horizontal_edges(0.1)[0]
            
            
#            width = lf1.boundary_segments[0].length
            width_p = pv_seg.length
            dist = width_p
            t_vec = Vector3D(dist,0,0)
#            if i < 2:
            lf2 = lf1.move(t_vec)
            
            face_list.append(rfg.from_face3d(lf2))
#            if i == 2:
#                break


    return face_list

def image_plane(faces, w, h):
    lb_m = []
    x_interval = 100
    y_interval = 50

    lb_s = []
    for i, f in enumerate(faces):
        fm = rtg.to_face3d(f)[0].triangulated_mesh3d
        ls = rtg.to_face3d(f)[0]
        
        lb_m.append(fm)
        lb_s.append(ls)
        
    bm = lb_m[0].join_meshes(lb_m)

    ip_face = bm.faces[0]

#    ll = rtg.to_face3d(faces[0])[0].min
#    ur = rtg.to_face3d(faces[-1])[0].max
#    ul = Point3D(ll[0], 0, ur[2])
#    lr = Point3D(ur[0], 0, ll[2])
#
#    wdist = ll.distance_to_point(lr)
#    hdist = ll.distance_to_point(ul)

    ll = rtg.to_face3d(faces[0])[0].min
    ur = Point3D(ll[0] + w, ll[1], ll[2] + h)
    ul = Point3D(ll[0], ll[1], ll[2] + h)
    lr = Point3D(ll[0] + w, ll[1], ll[2])


    x_spacing = w / x_interval 
    y_spacing = h / y_interval

    gs = Face3D([ll, lr, ur,ul])
#    gs = gs.sub_faces_by_ratio(.05)
#    print(gs)
#    return(gs)
    j_srf = Face3D([ll, ul, ur,lr]).mesh_grid(x_dim=x_spacing)
    
    
    cntrs = j_srf.face_centroids
    verts = j_srf.vertices
#    return [from_point3d(ll), from_point3d(lr), from_point3d(ur), from_point3d(ul)]
    
    return [rfg.from_mesh3d(j_srf), cntrs, verts]

def get_aligned_normals(m):
    v = [from_vector3d(rtg.to_face3d(face)[0].normal) for face in m]
    p = [from_point3d(rtg.to_face3d(face)[0].center) for face in m]
    
    return [p, v]
    
    
def get_bottom(poly):

    for i, f in enumerate(poly.faces):
        if f.normal[2] == -1.0:
            return f

def transpose_pts(pts):

    p1 = None
    matrix = []
    rows = 0
    for i, p in enumerate(pts):
        if i == 0:
            p1 = p
            rows += 1
        else:
            if p[2] != p1[2]:
                rows+=1
            else:
                break
    c=0
    while c<len(pts):
      matrix.append(pts[c:c+rows])
      c+=rows

    matrix_t = [list(i) for i in zip(*matrix)]
    return matrix, matrix_t

def bin_pts(faces, pts, orts):
    rows = len(pts)
    cols = len(pts[0])
#    print(rows, cols)
    
    label_mtx = []
    for i in range(rows):
        tmp = []
        for j in range(cols):
            tmp.append(99)
        label_mtx.append(tmp)
        
    bins = [[] for face in faces]
    
    for i in range(rows):
        for j in range(cols):
            inside = False
            for k, f in enumerate(faces):
                lb_f = rtg.to_face3d(f)[0]
#                print(pts[i][j])
                lb_pt = pts[i][j] #rtg.to_point3d()
                inside = lb_f.is_point_on_face(lb_pt, 0.001)
                
                if inside:
                    
                    bins[k].append(from_point3d(lb_pt))
                    label_mtx[i][j] = orts[k]

        
    return bins, label_mtx



def get_dims(_hb_objs):
#        bot_srfs = []
    perims = []
    heights = []
    for m in _hb_objs:
        model = m
        room = model.rooms[0]
        polyface3d = room.geometry
        _h = polyface3d.max[2] - polyface3d.min[2]
        
        heights.append(_h)

        bot_face = get_bottom(polyface3d)
        perims.append(bot_face.perimeter)

    max_w = max(perims)
    max_h = max(heights)
    
    return max_w, max_h
def get_facade_mask(hb_objs, w, h):
    if hb_objs is not None:
    
        model = hb_objs
        room = model.rooms[0]
    
        polyface3d = room.geometry
        _h = polyface3d.max[2] - polyface3d.min[2]
        
        pf = []
        for i in polyface3d:
            pf.append(i)

        
        bot_face = get_bottom(polyface3d)
        
        polyface = bot_face.flip()
        
        extr = rfg.from_face3d_to_solid(polyface, _h)

        #Get in place faces, normals, center, oriention
        f,p,v, o, o_index = get_normals(extr)
        
        # Align Facades
        aligned_faces = align(f, v)
        m = aligned_faces
        #Get aligned center and normals
    #    p, v = get_aligned_normals(m)
        
        #Get image plane overlaid on all facades
        ip, fcnts, verts = image_plane(m, w, h)
        
        mtx, mtxt = transpose_pts(fcnts)
        
        tm = rtg.to_face3d(m[0])[0]
        
        fac_bins, orient_mtx = bin_pts(m, mtxt, o_index)
        
#        check_ind = 0
#        lab1 = orient_mtx[check_ind]
#        pt1 = [rfg.from_point3d(_p) for _p in mtxt[check_ind]]
# 
        
    #    bhg = rtg.to_point3d(bin_pts(m, mtxt, or))
    #    bo = tm.is_point_on_face(bhg, 0.001)
    #    print(bo)
        return [lab1,lab2,lab3,lab4,lab5,pt1,pt2,pt3,pt4, pt5, f,o,p, m,ip]
        

#Get max dims
w, h = get_dims(_hb_objs)

#Extract Orientation Image/Mask
f ,o, p,m,ip  = get_facade_mask(_hb_objs[1], w, h)