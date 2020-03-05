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

fit_corner - Finds the equations of 2 perpendicular lines that fit an unlabelled
    set of points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found by
    minimising least squares error.

fit_corner2 - Finds the equations of 2 perpendicular lines that fit an
    unlabelled set of points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found
    by RANSAC and least squares.
"""

import open3d as o3d
import numpy as np
import cmath
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
from scipy import special
from itertools import combinations
from copy import deepcopy


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


def fit_corner(pts, max_iter=30, tol=0.01):
    """
    Finds the equations of 2 perpendicular lines that fit an unlabelled set of
    points such that x1^Tn=1 & x2^TRn/alpha = 1. Fit is found by minimising
    least squares error.

    Parameters
    ----------
    pts : np.array(m,2)
        set of m 2d points to be fit
    max_iter : int
        Maximum number of iterations to carry out the fitting
    tol : float < 1
        Maximum percentage change in parameters in each loop before breaking
        iteration early

    Returns
    -------
    n : np.array(2)
        2d array for vector equation
    alpha : float
        scalar for perpendicular line.
    """
    # put intitial intersection at centroid
    mean_pts = np.mean(pts, axis=0)
    R = np.array([[0, -1],  # 90 degree rotation matrix
                  [1, 0]])

    # find initial line using RANSAC and align it to pass through centroid
    x = pts[:, 0].reshape(-1, 1)
    y = pts[:, 1].reshape(-1, 1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x.reshape(-1, 1), y)
    n = np.array([ransac.estimator_.coef_, ransac.estimator_.intercept_])
    n = n.flatten()
    n = np.array([-n[0]/n[1], 1/n[1]])
    n /= np.linalg.norm(n)
    dist = n.dot(mean_pts)  # scale to pass through points
    n /= dist
    alpha = R.dot(n).dot(mean_pts)  # alpha scales second line

    n1 = n
    n2 = R.dot(n)/alpha
    for i in range(max_iter):
        # calculate distance to lines
        l1 = np.abs(pts.dot(n1)-1)/np.linalg.norm(n1)
        l2 = np.abs(pts.dot(n2)-1)/np.linalg.norm(n2)

        # get indexes closest to each line
        ind1 = np.where(l1 < l2)[0]
        ind2 = np.where(l1 > l2)[0]

        x1s = pts[ind1]
        x2s = pts[ind2]

        n_old = n
        n = np.sum(np.linalg.pinv(x1s), axis=1)

        alpha_old = alpha
        alpha = np.mean(x2s.dot(R.dot(n)))
        n1 = n
        n2 = R.dot(n)/alpha
        if abs(alpha_old-alpha)/alpha_old < tol and  \
                abs(np.linalg.norm(n-n_old))/np.linalg.norm(n_old) < tol:
            break

    plt.scatter(x1s[:, 0], x1s[:, 1])
    plt.scatter(x2s[:, 0], x2s[:, 1])
    return n, alpha


def fit_corner2(pts, max_iter=500, tol=0.01, n=None, alpha=None):
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
    tol : float < 1
        Maximum percentage change in parameters in each loop before breaking
        iteration early

    Returns
    -------
    n : np.array(2)
        2d array for vector equation
    alpha : float
        scalar for perpendicular line.
    """
    # define utility function for getting ransac results
    def fit_ransac(x1s, ransac):
        x1 = x1s[:, 0].reshape(-1, 1)
        y1 = x1s[:, 1].reshape(-1, 1)
        ransac.fit(x1, y1)
        inliers1 = x1s[ransac.inlier_mask_]
        n = np.array([ransac.estimator_.coef_, ransac.estimator_.intercept_])
        n = n.flatten()
        n = np.array([-n[0]/n[1], 1/n[1]])
        return n, inliers1

    # put intitial intersection at centroid
    mean_pts = np.mean(pts, axis=0)
    R = np.array([[0., -1.],  # 90 degree rotation matrix
                  [1., 0.]])

    ransac = linear_model.RANSACRegressor()
    if n is None:
        top = True
        n = np.random.rand(2)  # arbitrary starting vector
        n /= np.linalg.norm(n)
        dist = n.dot(mean_pts)  # scale to pass through points
        n /= dist
        alpha = R.dot(n).dot(mean_pts)  # alpha scales second line
    else:
        top = False

    n1 = n
    n2 = R.dot(n)/alpha
    for i in range(max_iter):
        # calculate distance to lines
        l1 = np.abs(pts.dot(n1)-1)/np.linalg.norm(n1)
        l2 = np.abs(pts.dot(n2)-1)/np.linalg.norm(n2)

        # get indexes closest to each line
        ind1 = np.where(l1 < l2)[0]
        ind2 = np.where(l1 > l2)[0]

        x1s = pts[ind1]
        x2s = pts[ind2]

        n_old = n

        try:
            n, inliers1 = fit_ransac(x1s, ransac)
            if len(x2s) < 3:
                x2s =np.append(x2s, pts[ind1[np.logical_not(ransac.inlier_mask_)]], axis=0)
        except ValueError:
            pass

        alpha_old = alpha
        n1 = n
        alpha, inliers2in = ransac1d(x2s.dot(R.dot(n)), 3, 50)
        inliers2 = x2s[inliers2in]
        # error would have been raised above if this is true so new n has not
        # been calculated
        if len(x1s) < 3:
            x1s = np.append(x1s, np.delete(x2s, inliers2in, 0), axis=0)
            if len(x1s) >= 3:
                n, inliers1 = fit_ransac(x1s, ransac)

        if abs(alpha_old-alpha)/alpha_old < tol and  \
                abs(np.linalg.norm(n-n_old))/np.linalg.norm(n_old) < tol:
            break
        n2 = R.dot(n)/alpha

    c1 = inliers1.dot(n) - 1
    c2 = inliers2.dot(R.dot(n))/alpha - 1
    cost = c1.T.dot(c1) + c2.T.dot(c2)

    # since solution is not symmetric, repeat with optimal n rotated and pick
    # best fit
    if top:
        n_new, alpha_new, i, j, cost2 = fit_corner2(pts, n=R.dot(n)/alpha, alpha=np.linalg.norm(n)/alpha)
        if cost2 < cost:
            n = n_new
            alpha = alpha_new
            ind1 = i
            ind2 = j
            cost = cost2

        # set up return convention
        return_set = [(n, alpha, ind1, ind2, cost),
                      (R.dot(n)/alpha, -1/alpha, ind2, ind1, cost)]

        n2 = R.dot(n)/alpha
        intersection = np.linalg.inv(np.array([n, n2])).dot(np.ones(2))
        pts_copy = deepcopy(pts)
        pts_copy -=intersection
        ang = np.zeros(2)
        mean1 = np.mean(pts_copy[ind1], axis=0)
        mean2 = np.mean(pts_copy[ind2], axis=0)
        ang[0] = cmath.polar(complex(*mean1))[1]
        ang[1] = cmath.polar(complex(*mean2))[1]

        # origin is internal to the building
        if ang.min() > 0 or ang.max() < 0 or ang.max() - ang.min() > 1.01*np.pi/2:
            for i in range(2):
                if ang[i] < 0:
                    ang[i] += 2*np.pi

        return return_set[np.argmin(ang)]


    return n, alpha, ind1, ind2, cost


def ransac1d(pts, min_samp, iter):
    """
    Returns the mean of the points calculated using RANSAC

    Parameters
    ----------
    pts : np.array
        1d array of data points
    min_samp : int
        Minimum number of points to calculate the mean from
    iter : int
        Number of iterations to test for.

    Returns
    -------
    mean : float
        The mean of the inliers in the data.
    inliers : np.array(int)
        Indicies of the inliers in the input points.
    """
    score_best = np.inf

    if special.comb(len(pts), min_samp, exact=True) < iter:
        combs = np.array(combinations(pts, min_samp))
        samples = np.mean(combs)
    else:
        samples = np.mean(np.random.choice(pts, (min_samp, iter)), axis=0)

    mads = stats.median_absolute_deviation(pts)
    for alpha in samples:
        residuals = np.abs(pts - alpha)
        inliers = np.where(residuals < 3*mads)[0]
        if len(inliers)/len(pts) < 0.5:
            continue
        score = np.linalg.norm(residuals[inliers])**2
        if score < score_best:
            score_best = score
            inliers_best = inliers

    try:
        return np.mean(pts[inliers_best]), inliers
    except UnboundLocalError:
        return np.mean(pts), []

def open_refs():
    model_titles = ["Fireblade", "Fire Warrior", "Commander", "Broadside"]
    models = {}
    model_clouds = {}
    print("Loading References...")
    for model in model_titles:
        tms = o3d.io.read_triangle_mesh("./Point Clouds/Photogram Refs/{}/texturedMesh.obj".format(model))
        R = np.load("./Point Clouds/Photogram Refs/{}/R_scaling.npy".format(model))
        tms.transform(R)
        print(R[-1,-1])
        models[model] = tms
        model_clouds[model] = o3d.geometry.PointCloud(tms.vertices)
        model_clouds[model].estimate_normals()

    return models, model_clouds
