"""
Utility functions for displaying, formatting and colouring point clouds

Author O.G.Jones

Functions
---------
hist_normals - Returns a list of most common normals ordered by magnitude for a
    given point cloud. All vectors returned are normalised by the largest
    magnitude.

clean_cloud - Cleans a point supplied point cloud of statistical outliers.

display_inlier_outlier - Displays the point cloud with points of index in ind
    highlighted red

create_vector_graph - Creats a Line Geometry of the supplied vectors coming
    from the origin.

crop_distance - Crops a cloud to everything less than <region> from the
    audience.

create_origin_plane - Creates plane aligned with origin and normal in y
    direction.
"""

import open3d as o3d
import numpy as np


def hist_normals(normals, bin_size=0.95):
    """
    Returns a list of most common normals ordered by magnitude for a given point
    cloud. All vectors returned are normalised by the largest magnitude.

    Parameters
    ----------
    pcd: open3d.geometry.PointCloud
        Point cloud to be histogrammed.

    bin_size: float (=0.95)
        tolerance of the dot product to accept a match.

    Returns
    -------
    hist_vect: np.array/ open3d.LineSet
    """
    # create empty list to bin into
    vec_list = []
    mags = 1/np.linalg.norm(normals, axis=1)
    unit_norms = np.array([i*j for i, j in zip(normals, mags)])
    while len(normals) > 0:
        # select first normal and get dot product with all other normals
        normal = unit_norms[0]
        dot_prod = unit_norms.dot(normal)
        ind = np.where(np.abs(dot_prod) > bin_size)[0]
        # dot with sign incase vectors are aligned but in opposite directions
        vec = np.sign(dot_prod[ind]).dot(normals[ind])
        vec_list.append(vec)
        # delete all normals that have been matched
        normals = np.delete(normals, ind, axis=0)
        unit_norms = np.delete(unit_norms, ind, axis=0)

    vec_list = np.array(vec_list)
    # normalise vectors based on largest magnitude
    mags = np.max(np.linalg.norm(vec_list, axis=0))
    vec_list /= mags

    return vec_list


def clean_cloud(pcd_fp, std_ratio=1, nb_neighbors=20, view=False, pcd_sl=None):
    """
    Cleans a point supplied point cloud of statistical outliers.

    Parameters
    ----------
    pcd_fp : str/ open3d.geometry.PointCloud
        File path to point cloud data or already opened point cloud data.
    std_ratio : float (1.0)
        Standard Ratio for statistical outlier removal
    nb_neighbors : int (20)
        Number of nearest neighbours to consider.
    view : bool (False)
        If True the point cloud will be visualised with outliers highlighted.
    pcd_sl : str (None)
        File path of where to save the cleaned cloud. If None, the cloud is not
        saved.

    Returns
    -------
    cl : open3d.geometry.PointCloud
        Cleaned point cloud
    ind : numpy.array (int)
        Indexes in original cloud of
    """

    if type(pcd_fp) == str:
        pcd = o3d.io.read_point_cloud(pcd_fp)
    else:
        pcd = pcd_fp

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    if view:
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
        display_inlier_outlier(pcd, ind)

    if pcd_sl is not None:
        o3d.io.write_point_cloud(pcd_sl, cl)

    return cl, ind


def display_inlier_outlier(cloud, ind):
    """
    Displays the point cloud with points of index in ind highlighted red

    Parameters
    ----------
    cloud : open3d.geometry.PointCloud
        cloud from which outliers have been found
    ind : numpy.array (int)
        Indexes in original cloud of outliers
    """
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return


def create_vector_graph(vectors):
    """
    Creats a Line Geometry of the supplied vectors coming from the origin.

    Parameters
    ----------
    vectors : list of 3d vectors
        List of vectors to form geometry from

    Returns
    -------
    line_set : o3d.geometry.LineSet()
        Geometry of vectors to be displayed
    """
    # create points with 0 vector
    points = np.append(np.zeros((1, 3)), vectors, axis=0)

    # create set of lines each starting from 0,0,0
    lines = np.array([[0, i+1]for i in range(len(vectors))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def crop_distance(pcd, region):
    """Crops a cloud to everything less than <region> from the audience"""
    points = np.asarray(pcd.points)
    dist = np.linalg.norm(points, axis=1)
    ind = [i for i, j in enumerate(dist) if j > region]
    return ind


def create_origin_plane(size):
    """Creates plane aligned with origin and normal in y direction"""
    plane = o3d.geometry.PointCloud()
    x = np.linspace(-0.1, 0.1, size)
    y = np.linspace(-0.1, 0.1, size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    plane_points = np.array([1, 0, 0])*np.reshape(xv, (len(xv), 1)) + \
        np.array([0, 0, 1])*np.reshape(yv, (len(yv), 1))
    plane.points = o3d.utility.Vector3dVector(plane_points)
    plane.paint_uniform_color([0, 0.5, 0])
    return plane


def align_vectors(origin, target):
    """
    Determines the transformation to map one vector onto another.

    Parameters
    ----------
    origin : np.array(3)
        3D vector that is unaligned
    target : np.array(3)
        3D vector onto which the transformation will map <origin>

    Returns
    -------
    T: np.array(4,4)
        4x4 homogenous transformation matrix that rotates origin onto target.
    """

    cross = -np.cross(target, origin)
    cos_ang = target.dot(origin)

    cross_skew = np.array([[0,         -cross[2], cross[1]],
                           [cross[2],  0,         -cross[0]],
                           [-cross[1], cross[0],  0]])

    R = np.identity(3) + cross_skew + np.matmul(cross_skew, cross_skew) * \
        (1-cos_ang)/(np.linalg.norm(cross)**2)

    R = np.array([[R[0][0], R[0][1], R[0][2], 0],
                  [R[1][0], R[1][1], R[1][2], 0],
                  [R[2][0], R[2][1], R[2][2], 0],
                  [0,       0,       0,       1]])
    return R


def colour_labels(pcd, labels):
    """
    Colours the points in a point cloud corresponding to their labels.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to be coloured.

    labels : numpy.array(int)
        Integer labels for each point in pcd.

    Returns
    -------
    pcd : open3d.geometry.PointCloud
        Coloured point cloud
    """

    colours = np.random.rand(labels.max()+1-labels.min(), 3)
    for i in range(len(labels)):
        pcd.colors[i] = colours[labels[i]]

    return pcd

def bounding_box_2d(points):
    a  = np.array([(3.7, 1.7), (4.1, 3.8), (4.7, 2.9), (5.2, 2.8), (6.0,4.0), (6.3, 3.6), (9.7, 6.3), (10.0, 4.9), (11.0, 3.6), (12.5, 6.4)])
    ca = np.cov(a,y = None,rowvar = 0,bias = 1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)



    #use the inverse of the eigenvectors as a rotation matrix and
    #rotate the points so they align with the x and y axes
    ar = np.dot(a,np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5

    # the center is just half way between the min and max xy
    center = mina + diff

    #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])

    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)

    return
