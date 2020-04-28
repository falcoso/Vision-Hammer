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

fit_corner - Finds the equations of 2 perpendicular lines that fit an
    unlabelled set of points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found
    by RANSAC and least squares.
"""

import open3d as o3d
import numpy as np
import scipy as sp
import cmath

from scipy import stats
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt


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
        dot_prod = unit_norms @ normal
        ind = np.where(np.abs(dot_prod) > bin_size)[0]
        # dot with sign incase vectors are aligned but in opposite directions
        vec = np.sign(dot_prod[ind]) @ normals[ind]
        vec_list.append(vec)
        # delete all normals that have been matched
        normals = np.delete(normals, ind, axis=0)
        unit_norms = np.delete(unit_norms, ind, axis=0)

    vec_list = np.array(vec_list)
    # normalise vectors based on largest magnitude
    mags = np.max(np.linalg.norm(vec_list, axis=0))
    vec_list /= mags
    for i in range(len(vec_list)):
        vec_list[i] *= np.sign(vec_list[i, np.argmax(abs(vec_list[i]))])

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


def create_origin_plane(no_pts, radius):
    """Creates plane aligned with origin and normal in y direction"""
    plane = o3d.geometry.PointCloud()
    x = np.linspace(-radius, radius, no_pts)
    y = np.linspace(-radius, radius, no_pts)
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
    cos_ang = target @ origin

    cross_skew = np.array([[0,         -cross[2],  cross[1]],
                           [cross[2],  0,         -cross[0]],
                           [-cross[1], cross[0],         0]])

    R = np.identity(3) + cross_skew + np.matmul(cross_skew, cross_skew) * \
        (1-cos_ang)/(np.linalg.norm(cross)**2)

    T = np.identity(4)

    T[:-1, :-1] = R
    return T


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
    colours[-1] = np.array([0, 0, 0])
    for i in range(len(labels)):
        pcd.colors[i] = colours[labels[i]]

    return pcd


def fit_corner(pts, max_iter=50, tol=0.01, n=None, alpha=None):
    """
    Finds the equations of 2 perpendicular lines that fit an unlabelled set of
    points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found by RANSAC and
    least squares.

    Parameters
    ----------
    pts : np.array(m,2)
        set of m 2d points to be fit
    max_iter : int
        Maximum number of iterations to carry out the fitting
    tol : float +ve < 1
        Maximum percentage change in parameters in each loop before breaking
        iteration early

    Returns
    -------
    n : np.array(2)
        2d array for vector equation
    alpha : float
        scalar for perpendicular line.
    inliers1 : np.array(int)
        Inlier points that fit to pts @ n =1
    inliers2 : np.array(int)
        Inlier points that fit to pts @ R @ n/alpha = 1
    cost : float
        Final value of the Geometric Least Squares cost function
    """
    # put intitial intersection at centroid
    R = np.array([[0., -1.],  # 90 degree rotation matrix
                  [1., 0.]])

    R45 = np.array([[1., -1.],  # 45 degree rotation matrix
                    [1., 1.]])
    R45 *= 1/np.sqrt(2)

    n_thresh = 1
    n_min_samp = 2
    n_iter = 1000
    alpha_thresh = None
    alpha_min_samp = 3
    alpha_iter = 500
    mode = 1

    if n is None:
        top = True
        # start off by fitting RANSAC line to all points
        n, inliers1in = ransac2d(pts, n_min_samp, n_iter, n_thresh)

        # use only the outliers to initialise alpha
        x2s = np.delete(pts, inliers1in, 0)
        alpha, inliers2in = ransac1d(x2s @ R @ n/np.linalg.norm(n),
                                     alpha_min_samp, alpha_iter, alpha_thresh)
        alpha *= np.linalg.norm(n)
    else:
        top = False

    # MAIN FUNCTION LOOP
    n_old = np.ones((2, 2))*0.01  # make small so exit isn't done on first step
    alpha_old = np.ones(2)*0.01
    for i in range(max_iter):
        # bisect points between corner
        n2 = (R @ n)/alpha
        c = np.linalg.inv(np.array([n, n2])) @ np.ones(2)
        ind1 = []
        ind2 = []
        bisec = R45 @ n
        for i in range(2):
            if i != 0:
                bisec = R45.T @ n

            dot_match = pts @ bisec - c @ bisec
            ind1.append(np.where(dot_match > 0)[0])
            ind2.append(np.where(dot_match < 0)[0])

        lengths = np.array([abs(len(ind1[i])-len(ind2[i])) for i in range(2)])
        ind1 = ind1[np.argmin(lengths)]
        ind2 = ind2[np.argmin(lengths)]
        x1s = pts[ind1]
        x2s = pts[ind2]

        n_old[1] = n_old[0]
        n_old[0] = n

        # fit n
        if len(x1s) >= n_min_samp:
            n, inliers1in = ransac2d(x1s, n_min_samp, n_iter, n_thresh)
            inliers1 = x1s[inliers1in]
            if len(x2s) < 3:
                x2s = np.append(x2s, np.delete(x1s, inliers1in, 0), axis=0)

        alpha_old[1] = alpha_old[0]
        alpha_old[0] = alpha
        alpha, inliers2in = ransac1d(x2s @ R @ n/np.linalg.norm(n),
                                     alpha_min_samp, alpha_iter, alpha_thresh)
        alpha *= np.linalg.norm(n)
        inliers2 = x2s[inliers2in]

        # fit alpha
        if len(x1s) < n_min_samp:
            x1s = np.append(x1s, np.delete(x2s, inliers2in, 0), axis=0)
            if len(x1s) >= n_min_samp:
                n, inliers1in = ransac2d(x1s, n_min_samp, n_iter, n_thresh)
                inliers1 = x1s[inliers1in]
            else:
                raise RuntimeError("Unable to assign enough points to the line")

        # exit conditions
        if abs(alpha_old[0]-alpha)/alpha_old[0] < tol and  \
                abs(np.linalg.norm(n-n_old[0]))/np.linalg.norm(n_old[0]) < tol:
            break
        elif abs(alpha_old[1]-alpha)/alpha_old[1] < tol and  \
                abs(np.linalg.norm(n-n_old[1]))/np.linalg.norm(n_old[1]) < tol:
            if mode == 0:
                mode = 1
            else:
                break

    # calculate costs on final values
    c1 = inliers1 @ n - 1
    c2 = (inliers2 @ R @ n)/alpha - 1
    cost = (c1.T @ c1)/len(inliers1) + (c2.T @ c2)/len(inliers2)

    # since solution is not symmetric, repeat with optimal n rotated and pick
    # best fit
    if top:
        n_new, alpha_new, i, j, cost2 = fit_corner(pts, n=(R @ n)/alpha,
                                                   alpha=np.linalg.norm(n)/alpha)
        if cost2 < cost:
            n = n_new
            alpha = alpha_new
            inliers1 = i
            inliers2 = j
            cost = cost2

        # set up return convention
        return_set = [(n, alpha, inliers1, inliers2, cost),
                      ((R @ n)/alpha, -1/alpha, inliers2, inliers1, cost)]

        n2 = (R @ n)/alpha
        intersection = np.linalg.inv(np.array([n, n2])) @ np.ones(2)
        ang = np.zeros(2)
        mean1 = np.mean(inliers1-intersection, axis=0)
        mean2 = np.mean(inliers2-intersection, axis=0)
        ang[0] = cmath.polar(complex(*mean1))[1]
        ang[1] = cmath.polar(complex(*mean2))[1]

        # origin is internal to the building
        if ang.min() > 0 or ang.max() < 0 or ang.max() - ang.min() > 1.01*np.pi/2:
            for i in range(2):
                if ang[i] < 0:
                    ang[i] += 2*np.pi

        return return_set[np.argmin(ang)]

    return n, alpha, inliers1, inliers2, cost


def fit_corner2(pts, max_iter=50, tol=0.01, n=None, alpha=None):
    """
    Finds the equations of 2 perpendicular lines that fit an unlabelled set of
    points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found by RANSAC and
    least squares.

    Parameters
    ----------
    pts : np.array(m,2)
        set of m 2d points to be fit
    max_iter : int
        Maximum number of iterations to carry out the fitting
    tol : float +ve < 1
        Maximum percentage change in parameters in each loop before breaking
        iteration early

    Returns
    -------
    n : np.array(2)
        2d array for vector equation
    alpha : float
        scalar for perpendicular line.
    inliers1 : np.array(int)
        Inlier points that fit to pts @ n =1
    inliers2 : np.array(int)
        Inlier points that fit to pts @ R @ n/alpha = 1
    cost : float
        Final value of the Geometric Least Squares cost function
    """
    # put intitial intersection at centroid
    R = np.array([[0., -1.],  # 90 degree rotation matrix
                  [1., 0.]])

    R45 = np.array([[1., -1.],  # 45 degree rotation matrix
                    [1., 1.]])
    R45 *= 1/np.sqrt(2)

    n_thresh = 1
    n_min_samp = 2
    n_iter = 1000
    alpha_thresh = 1
    alpha_min_samp = 3
    alpha_iter = 500
    d = np.zeros(2)
    d_old = np.zeros(2)
    n_old = np.zeros(2)
    cost_old = np.zeros(2)

    # start off by fitting RANSAC line to all points
    n, _ = ransaceig(pts-np.mean(pts, axis=0), n_min_samp, n_iter, n_thresh)
    d[0], _ = ransac1d(pts @ n, alpha_min_samp, alpha_iter, alpha_thresh)
    d[1], _ = ransac1d(pts @ R @ n, alpha_min_samp, alpha_iter, alpha_thresh)

    # MAIN FUNCTION LOOP
    for i in range(max_iter):
        # bisect points between corner
        n2 = (R @ n)
        c = np.linalg.inv(np.array([n, n2])) @ d
        ind1 = []
        ind2 = []
        bisec = R45 @ n
        for i in range(2):
            if i != 0:
                bisec = R45.T @ n

            dot_match = pts @ bisec - c @ bisec
            ind1.append(np.where(dot_match > 0)[0])
            ind2.append(np.where(dot_match < 0)[0])

        lengths = np.array([abs(len(ind1[i])-len(ind2[i])) for i in range(2)])
        ind1 = ind1[np.argmin(lengths)]
        ind2 = ind2[np.argmin(lengths)]
        x1s = pts[ind1]
        x2s = pts[ind2]

        # fit n
        xs = np.append(x1s-np.median(x1s, axis=0), (x2s-np.median(x2s, axis=0)) @ R, axis=0)
        n, _ = ransaceig(xs, n_min_samp, n_iter, n_thresh)
        d[0], inliers1 = ransac1d(x1s @ n, alpha_min_samp, alpha_iter, alpha_thresh)
        d[1], inliers2 = ransac1d(x2s @ R @ n, alpha_min_samp, alpha_iter, alpha_thresh)

        inliers1 = x1s[inliers1]
        inliers2 = x2s[inliers2]
        c1 = inliers1 @ n - d[0]
        c2 = inliers2 @ R @ n - d[1]
        cost = (c1.T @ c1)/len(inliers1) + (c2.T @ c2)/len(inliers2)

        if abs(cost - cost_old[0]) < cost_old[0]*tol:
            break
        elif abs(cost - cost_old[1]) < cost_old[1]*tol:
            if cost_old[0] < cost:
                d = d_old
                n = n_old
            break

        cost_old[1] = cost_old[0]
        cost_old[0] = cost
        d_old = d
        n_old = n

    # set up return convention
    return_set = [(n, d, inliers1, inliers2, cost),
                  (R @ n, d @ R, inliers2, inliers1, cost)]

    n2 = (R @ n)
    intersection = np.linalg.inv(np.array([n, n2])) @ d
    ang = np.zeros(2)
    mean1 = np.mean(inliers1-intersection, axis=0)
    mean2 = np.mean(inliers2-intersection, axis=0)
    ang[0] = np.arctan2(*mean1[::-1])
    ang[1] = np.arctan2(*mean2[::-1])

    # origin is internal to the building
    if ang.min() > 0 or ang.max() < 0 or ang.max() - ang.min() > 1.01*np.pi/2:
        for i in range(2):
            if ang[i] < 0:
                ang[i] += 2*np.pi

    return return_set[np.argmin(ang)]


def fit_corner3(pts, max_iter=50, tol=0.01, n=None, alpha=None):
    """
    Finds the equations of 2 perpendicular lines that fit an unlabelled set of
    points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found by RANSAC and
    least squares.

    Parameters
    ----------
    pts : np.array(m,2)
        set of m 2d points to be fit
    max_iter : int
        Maximum number of iterations to carry out the fitting
    tol : float +ve < 1
        Maximum percentage change in parameters in each loop before breaking
        iteration early

    Returns
    -------
    n : np.array(2)
        2d array for vector equation
    alpha : float
        scalar for perpendicular line.
    inliers1 : np.array(int)
        Inlier points that fit to pts @ n =1
    inliers2 : np.array(int)
        Inlier points that fit to pts @ R @ n/alpha = 1
    cost : float
        Final value of the Geometric Least Squares cost function
    """
    # put intitial intersection at centroid
    R = np.array([[0., -1.],  # 90 degree rotation matrix
                  [1., 0.]])

    R45 = np.array([[1., -1.],  # 45 degree rotation matrix
                    [1., 1.]])
    R45 *= 1/np.sqrt(2)

    n_thresh = 1
    n_min_samp = 2
    n_iter = 1000
    alpha_thresh = 1
    alpha_min_samp = 3
    alpha_iter = 500
    d = np.zeros(2)
    d_old = np.zeros(2)
    n_old = np.zeros(2)
    cost_old = np.zeros(2)

    # start off by fitting RANSAC line to all points
    n, inliers = ransaceig(pts-np.mean(pts, axis=0), n_min_samp, n_iter, n_thresh)
    d[0], _ = ransac1d(pts @ n, alpha_min_samp, alpha_iter, alpha_thresh)
    d[1], _ = ransac1d(pts @ R @ n, alpha_min_samp, alpha_iter, alpha_thresh)

    # MAIN FUNCTION LOOP
    for i in range(max_iter):
        # bisect points between corner
        n2 = (R @ n)
        c = np.linalg.inv(np.array([n, n2])) @ d
        ind1 = []
        ind2 = []
        bisec = R45 @ n
        for i in range(2):
            if i != 0:
                bisec = R45.T @ n

            dot_match = pts @ bisec - c @ bisec
            ind1.append(np.where(dot_match > 0)[0])
            ind2.append(np.where(dot_match < 0)[0])

        lengths = np.array([abs(len(ind1[i])-len(ind2[i])) for i in range(2)])
        ind1 = ind1[np.argmin(lengths)]
        ind2 = ind2[np.argmin(lengths)]
        x1s = pts[ind1]
        x2s = pts[ind2]

        # fit n
        n, [inliers1, inliers2] = ransaceig3(x1s, x2s @ R, min_samp=n_min_samp, iter=n_iter, thresh=n_thresh)
        inliers1 = x1s[inliers1]
        inliers2 = x2s[inliers2]

        d[0] = np.mean(inliers1 @ n)
        d[1] = np.mean(inliers2 @ R @ n)

        c1 = inliers1 @ n - d[0]
        c2 = inliers2 @ R @ n - d[1]
        cost = (c1.T @ c1)/len(inliers1) + (c2.T @ c2)/len(inliers2)

        if abs(cost - cost_old[0]) < cost_old[0]*tol:
            break
        elif abs(cost - cost_old[1]) < cost_old[1]*tol:
            if cost_old[0] < cost:
                d = d_old
                n = n_old
            break

        cost_old[1] = cost_old[0]
        cost_old[0] = cost
        d_old = d
        n_old = n

    # set up return convention
    return_set = [(n, d, inliers1, inliers2, cost),
                  (R @ n, d @ R, inliers2, inliers1, cost)]

    n2 = (R @ n)
    intersection = np.linalg.inv(np.array([n, n2])) @ d
    ang = np.zeros(2)
    mean1 = np.mean(inliers1-intersection, axis=0)
    mean2 = np.mean(inliers2-intersection, axis=0)
    ang[0] = np.arctan2(*mean1[::-1])
    ang[1] = np.arctan2(*mean2[::-1])

    # origin is internal to the building
    if ang.min() > 0 or ang.max() < 0 or ang.max() - ang.min() > 1.01*np.pi/2:
        for i in range(2):
            if ang[i] < 0:
                ang[i] += 2*np.pi

    return return_set[np.argmin(ang)]


def ransac1d(pts, min_samp=1, iter=100, mads=None):
    """
    Returns the mean of the points calculated using RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int (=1)
        Minimum number of points to calculate the mean.
    iter : int (=100)
        Number of iterations to test for. If iterations exceed number of
        possible sample combinations then all combinations are tested.
    thresh : float (=None)
        Maximum deviation from the mean to still be classed as an inlier. If
        None, than the median_absolute_deviation is used as the threshold

    Returns
    -------
    mean : float
        The mean of the inliers in the data.
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    inliers_best = []

    if sp.special.comb(len(pts), min_samp, exact=True) < iter:
        samples = (np.mean(pts[list(i)]) for i in combinations(np.arange(len(pts)), min_samp))
    else:
        samples = np.mean(np.random.choice(pts, (iter, min_samp)), axis=1)

    if mads is None:
        mads = stats.median_absolute_deviation(pts)
    for alpha in samples:
        residuals = np.abs(pts - alpha)
        inliers = np.where(residuals < mads)[0]
        if len(inliers_best) < len(inliers):
            inliers_best = inliers

    try:
        return np.mean(pts[inliers_best]), inliers_best
    except UnboundLocalError:
        return np.mean(pts), np.arange(len(pts))


def ransac2d(pts, min_samp=2, iter=500, thresh=None):
    """
    Returns the line fitting the points using a Geometric Least Square RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int (=2)
        Minimum number of points to calculate the line from (minimum 2)
    iter : int (=100)
        Number of iterations to test for. If iterations exceed number of
        possible sample combinations then all combinations are tested.
    thresh : float (=None)
        maximum deviation from the line to still be classed as an inlier. If
        None, than the median_absolute_deviation is used as the threshold

    Returns
    -------
    n : np.array(2, dtype=float)
        The normal to the line such that pts @ n = 1
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    if min_samp < 2:
        raise ValueError("Minimum of 2 points required to fit a line")

    # generate samples
    if sp.special.comb(len(pts), min_samp, exact=True) < iter:
        samples = combinations(np.arange(len(pts)), min_samp)
    else:
        samples = np.random.choice(np.arange(len(pts)), (iter, min_samp))

    # set thresholding function
    if thresh is None:
        def mads(pts): return stats.median_absolute_deviation(pts)
    else:
        def mads(pts): return thresh

    inliers_best = []
    for sampleind in samples:
        sample = pts[list(sampleind)]
        n, _, rank, s = np.linalg.lstsq(sample, np.ones(sample.shape[0]),
                                        rcond=None)
        residuals = np.abs(pts @ n - 1)/np.linalg.norm(n)

        inliers = np.where(residuals < mads(residuals))[0]
        if len(inliers_best) < len(inliers):
            inliers_best = inliers

    if inliers_best != []:
        n, _, rank, s = np.linalg.lstsq(pts[inliers_best],
                                        np.ones(len(inliers_best)), rcond=None)
        return n, inliers_best
    else:
        n, _, rank, s = np.linalg.lstsq(pts, np.ones(pts.shape[0]), rcond=None)
        return n, np.arange(len(pts))


def ransaceig(pts, min_samp=2, iter=500, thresh=None):
    """
    Returns the line fitting the points using a Geometric Least Square RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int (=2)
        Minimum number of points to calculate the line from (minimum 2)
    iter : int (=100)
        Number of iterations to test for. If iterations exceed number of
        possible sample combinations then all combinations are tested.
    thresh : float (=None)
        maximum deviation from the line to still be classed as an inlier. If
        None, than the median_absolute_deviation is used as the threshold

    Returns
    -------
    n : np.array(2, dtype=float)
        The normal to the line such that pts @ n = 1
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    if min_samp < 2:
        raise ValueError("Minimum of 2 points required to fit a line")

    # generate samples
    if sp.special.comb(len(pts), min_samp, exact=True) < iter:
        samples = combinations(np.arange(len(pts)), min_samp)
    else:
        samples = np.random.choice(np.arange(len(pts)), (iter, min_samp))

    # set thresholding function
    if thresh is None:
        def mads(pts): return stats.median_absolute_deviation(pts)
    else:
        def mads(pts): return thresh

    inliers_best = []
    for sampleind in samples:
        sample = pts[list(sampleind)]
        sample_mean = np.mean(sample, axis=0)
        sample -= sample_mean
        s, w = np.linalg.eigh(sample.T @ sample)
        n = w[:, 0]
        residuals = np.abs((pts-sample_mean) @ n)

        inliers = np.where(residuals < mads(residuals))[0]
        if len(inliers_best) < len(inliers):
            inliers_best = inliers

    if inliers_best != []:
        pts_in = pts[inliers_best]
        pts_in -= np.mean(pts_in, axis=0)
        s, w = np.linalg.eigh(pts_in.T @ pts_in)
        n = w[:, 0]
        return n, inliers_best
    else:
        pts_mean = np.mean(pts, axis=0)
        pts_bar = pts-pts_mean
        s, w = np.linalg.eigh(pts_bar.T @ pts_bar)
        n = w[:, 0]
        return n, np.arange(len(pts))

def ransaceig2(x1s, x2s, min_samp=2, iter=500, thresh=None):
    """
    Returns the line fitting the points using a Geometric Least Square RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int (=2)
        Minimum number of points to calculate the line from (minimum 2)
    iter : int (=100)
        Number of iterations to test for. If iterations exceed number of
        possible sample combinations then all combinations are tested.
    thresh : float (=None)
        maximum deviation from the line to still be classed as an inlier. If
        None, than the median_absolute_deviation is used as the threshold

    Returns
    -------
    n : np.array(2, dtype=float)
        The normal to the line such that pts @ n = 1
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    if min_samp < 2:
        raise ValueError("Minimum of 2 points required to fit a line")

    # generate samples
    samples1 = np.random.choice(np.arange(len(x1s)), (iter, min_samp))
    samples2 = np.random.choice(np.arange(len(x2s)), (iter, min_samp))

    # set thresholding function
    if thresh is None:
        def mads(pts): return stats.median_absolute_deviation(pts)
    else:
        def mads(pts): return thresh

    inliers_best1 = []
    inliers_best2 = []
    for sampleind1, sampleind2 in zip(samples1, samples2):
        sample1 = x1s[list(sampleind1)]
        sample2 = x2s[list(sampleind2)]
        sample_mean1 = np.mean(sample1, axis=0)
        sample_mean2 = np.mean(sample2, axis=0)
        sample1 -= sample_mean1
        sample2 -= sample_mean2

        s, w = np.linalg.eigh(sample1.T @ sample1 + sample2.T @ sample2)
        n = w[:, 0]
        residuals1 = np.abs((x1s-sample_mean1) @ n)
        residuals2 = np.abs((x2s-sample_mean2) @ n)

        inliers1 = np.where(residuals1 < mads(residuals1))[0]
        if len(inliers_best1) < len(inliers1):
            inliers_best1 = inliers1

        inliers2 = np.where(residuals2 < mads(residuals2))[0]
        if len(inliers_best2) < len(inliers2):
            inliers_best2 = inliers2

    if inliers_best1 != [] and inliers_best2 !=[]:
        x1s_in = x1s[inliers_best1]
        x1s_in -= np.mean(x1s_in, axis=0)
        x2s_in = x2s[inliers_best2]
        x2s_in -= np.mean(x2s_in, axis=0)

        s, w = np.linalg.eigh(x1s_in.T @ x1s_in + x2s_in.T @ x2s_in)
        n = w[:, 0]
        return n, inliers_best1, inliers_best2
    else:
        x1s_in = x1s - np.mean(x1s,axis=0)
        x2s_in = x2s - np.mean(x2s,axis=0)
        s, w = np.linalg.eigh(x1s_in.T @ x1s_in + x2s_in.T @ x2s_in)
        n = w[:, 0]
        return n, np.arange(len(x1s)), np.arang(len(x2s))

def ransaceig3(*xs, min_samp=2, iter=500, thresh=None):
    """
    Returns the line fitting the points using a Geometric Least Square RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int (=2)
        Minimum number of points to calculate the line from (minimum 2)
    iter : int (=100)
        Number of iterations to test for. If iterations exceed number of
        possible sample combinations then all combinations are tested.
    thresh : float (=None)
        maximum deviation from the line to still be classed as an inlier. If
        None, than the median_absolute_deviation is used as the threshold

    Returns
    -------
    n : np.array(2, dtype=float)
        The normal to the line such that pts @ n = 1
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    if min_samp < 2:
        raise ValueError("Minimum of 2 points required to fit a line")

    # generate samples
    samples=[]
    for i in xs:
        samples.append(np.random.choice(np.arange(len(xs)), (iter, min_samp)))

    # set thresholding function
    if thresh is None:
        def mads(pts): return stats.median_absolute_deviation(pts)
    else:
        def mads(pts): return thresh

    inliers_best = [[]]*len(xs)
    for i in range(iter):
        sample_mean = []
        A = np.zeros((2,2))
        for j in range(len(samples)):
            sample = xs[j][list(samples[j][i])]
            sample_mean.append(np.mean(sample, axis=0))
            sample -= sample_mean[-1]
            A += sample.T @ sample

        s, w = np.linalg.eigh(A)
        n = w[:, 0]

        for j in range(len(samples)):
            residuals = np.abs((xs[j]-sample_mean[j]) @ n)
            inliers = np.where(residuals < mads(residuals))[0]
            if len(inliers_best[j]) < len(inliers):
                inliers_best[j] = deepcopy(inliers)

    A = np.zeros((2,2))
    for i in range(len(inliers_best)):
        if inliers_best[i] == []:
            inliers_best[i] = np.arange(len(xs[i]))
        x_in = xs[i][inliers_best[i]]
        x_in -= np.mean(x_in, axis=0)
        A += x_in.T @ x_in

    s, w = np.linalg.eigh(A)
    n = w[:, 0]
    return n, inliers_best


def _icp_correspondance(source, target_tree):
    """Picks correspondance under current transformation."""
    ids = -np.ones(len(source.points), dtype=int)
    for i in range(len(source.points)):
        k, id, _ = target_tree.search_knn_vector_3d(source.points[i], 1)
        ids[i] = np.asarray(id)[0]
    return ids


def _icp_get_inliers(xs, ts):
    """Filters points more than 2*mad from the median."""
    residuals = np.linalg.norm(xs-ts, axis=1)
    i_residuals = np.argsort(residuals)
    inliers = i_residuals[:int(0.8*len(residuals))]
    inliers = np.sort(inliers)
    return inliers


def icp_constrained(source, target, theta=0, iter=50, tol=0.01):
    """
    Point to point iterative closest point, constrained to rotate about the
    y-axis.

    Parameters
    ----------
    source : o3d.geometry.PointCloud
        Point cloud that will be rotated to best fit.
    target : o3d.geometry.PointCloud
        Point cloud to which source will be matched
    theta : float
        Initial rotation about y-axis to start fitting in radians.
    iter : int
        Maximum number of iterations.
    tol : float
        Percentage change in theta between iterations below which iteration
        stops.

    Returns
    -------
    R : np.array(4x4)
        4x4 transformation matrix.
    final_cost : float
        Cost per point of the given transformation.
    """
    target_tree = o3d.geometry.KDTreeFlann(target)
    cost = 0

    R_old = np.array([[np.cos(theta),  0, np.sin(theta), 0],
                      [0,              1,             0, 0],
                      [-np.sin(theta), 0, np.cos(theta), 0],
                      [0,              0,             0, 1]])
    source_c = deepcopy(source)
    source_c.transform(R_old)
    target_points = np.asarray(target.points)

    for i in range(iter):
        # get correspondance
        ids = _icp_correspondance(source_c, target_tree)

        # filter outliers
        xs = np.asarray(source_c.points)
        ts = target_points[ids]

        inliers = _icp_get_inliers(xs, ts)
        xs = xs[inliers]
        ts = ts[inliers]
        # trans = np.mean(xs-ts, axis=0)

        # remove y axis to constrain transformation about that axis
        mu_x = np.mean(xs, axis=0)
        mu_t = np.mean(ts, axis=0)

        xs_r = xs[:, ::2]  # removes y axis element
        ts_r = ts[:, ::2]

        R0 = (xs_r-mu_x[::2]).T @ (ts_r-mu_t[::2])
        u, s, v = np.linalg.svd(R0)
        R0 = u @ v

        R0 = np.array([[R0[0, 0], 0, R0[0, 1], 0],
                       [0,        1, 0,       0],
                       [R0[1, 0], 0, R0[1, 1], 0],
                       [0,        0, 0,       1]])

        trans = mu_t - mu_x @ R0[:3, :3].T
        R0[:3, -1] = trans
        source_c.transform(R0)
        R_old = R0 @ R_old

        p = xs-ts
        new_cost = np.sum(p @ p.T)/len(xs)

        if i > 2:
            if abs(cost-new_cost) < cost*tol:
                break

        cost = new_cost

    return R_old, cost


def icp_constrained_plane(source, target, theta=0, iter=50, tol=0.01):
    """
    Point to plane iterative closest point, constrained to rotate about the
    y-axis.

    Parameters
    ----------
    source : o3d.geometry.PointCloud
        Point cloud that will be rotated to best fit.
    target : o3d.geometry.PointCloud
        Point cloud to which source will be matched
    theta : float
        Initial rotation about y-axis to start fitting in radians.
    iter : int
        Maximum number of iterations.
    tol : float
        Minimum percentage change in cost between iterations.

    Returns
    -------
    R : np.array(4x4)
        4x4 transformation matrix.
    final_cost : float
        Cost per point of the given transformation.
    """
    def R(theta, trans=np.zeros(3)):
        T = np.array([[np.cos(theta),  0, np.sin(theta), 0],
                      [0,              1,             0, 0],
                      [-np.sin(theta), 0, np.cos(theta), 0],
                      [0,              0,             0, 1]])
        T[:3, -1] = trans
        return T

    target_tree = o3d.geometry.KDTreeFlann(target)

    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()

    R_old = R(theta)
    cost = 0

    source_c = deepcopy(source)
    source_c.transform(R_old)
    target_points = np.asarray(target.points)
    target_norms = np.asarray(target.normals)

    for i in range(iter):
        # get correspondance
        ids = _icp_correspondance(source_c, target_tree)

        # filter outliers
        xs = np.asarray(source_c.points)
        ts = target_points[ids]
        ns = target_norms[ids]

        inliers = _icp_get_inliers(xs, ts)
        xs = xs[inliers]
        ts = ts[inliers]
        ns = ns[inliers]

        # construct a matrix for least squares
        xs_r = xs[:, ::2]  # takes just x and z values and swaps them
        ns_r = ns[:, ::2]
        ns_r[:, -1] *= -1

        A = np.zeros((len(xs_r), 4))
        A[:, 0] = xs[:, -1]*ns[:, 0] - xs[:, 0]*ns[:, -1]
        A[:, 1:] = ns
        b = np.einsum('ij,ij->i', ts-xs, ns)
        lam, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        R0 = R(lam[0], lam[1:])
        source_c.transform(R0)
        R_old = R0 @ R_old

        new_cost = np.sum(np.abs(np.einsum('ij,ij->i', (xs-ts), ns)))/len(xs)
        if i > 2:
            if abs(cost-new_cost) < cost*tol:
                break

        cost = new_cost

    return R_old, cost


def fit_circle(pts):
    """
    Calculates the centre and radius of a set of points on a circle using a
    Kasa fit.

    Parameters
    ----------
    pts : np.array(N,2) float
        2D set of points on which to fit the circle.

    Returns
    -------
    c : np.array(2) float
        2D point locating the centre of the circle
    r : float
        Radius of the circle.
    """
    # add column of ones
    A = np.concatenate((pts, np.ones((pts.shape[0], 1))), 1)
    b = -np.einsum('ij,ij->i', pts, pts)
    lam, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    a = -lam[0]/2
    b = -lam[1]/2
    r = (lam[0]**2 + lam[1]**2 - 4*lam[2])/4
    r = np.sqrt(r)
    return np.array([a, b]), r
