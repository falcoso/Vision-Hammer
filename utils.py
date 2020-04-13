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
import cmath
from sklearn import linear_model
from scipy import stats
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


def fit_corner(pts, max_iter=500, tol=0.01, n=None, alpha=None):
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
    """
    # define utility function for getting ransac results
    def fit_ransac(x1s):
        ransac = linear_model.RANSACRegressor(min_samples=4)
        x1 = x1s[:, 0].reshape(-1, 1)
        y1 = x1s[:, 1].reshape(-1, 1)
        ransac.fit(x1, y1)
        n = np.array([ransac.estimator_.coef_, ransac.estimator_.intercept_])
        n = n.flatten()
        n = np.array([-n[0]/n[1], 1/n[1]])
        return n, np.where(ransac.inlier_mask_)[0]

    # put intitial intersection at centroid
    # mean_pts = np.mean(pts, axis=0)
    R = np.array([[0., -1.],  # 90 degree rotation matrix
                  [1., 0.]])

    R45 = np.array([[1., -1.],  # 90 degree rotation matrix
                    [1., 1.]])
    R45 *= 1/np.sqrt(2)

    if n is None:
        top = True
        # start off by fitting RANSAC line to all points
        n, inliers1 = fit_ransac(pts)
        alpha, inliers2in = ransac1d(pts @ R @ n/np.linalg.norm(n), 3, 50)
        alpha *= np.linalg.norm(n)
    else:
        top = False

    n2 = (R @ n)/alpha
    for i in range(max_iter):
        # calculate distance to lines
        # c = np.linalg.inv(np.array([n, n2])) @ np.ones(2)
        # bisec = R45 @ n
        # bisec /= np.linalg.norm(bisec)
        # dot_match = pts @ bisec - c @ bisec
        # ind1a = np.where(dot_match < 0)[0]
        # ind2a = np.where(dot_match > 0)[0]
        #
        # bisec = R45.T @ n
        # dot_match = pts @ bisec - c @ bisec
        # ind1b = np.where(dot_match < 0)[0]
        # ind2b = np.where(dot_match > 0)[0]
        # if abs(len(ind1a)-len(ind2a)) < abs(len(ind1b)-len(ind2b)):
        #     ind1 = ind1a
        #     ind2 = ind2a
        # else:
        #     print("Using other one?")
        #     ind1 = ind1b
        #     ind2 = ind2b
        l1 = (pts @ n - 1)/np.linalg.norm(n)
        l2 = (pts @ n2 - 1)/np.linalg.norm(n2)

        # get indexes closest to each line
        l1_abs = np.abs(l1)
        l2_abs = np.abs(l2)
        ind1 = np.where(l1_abs < l2_abs)[0]
        ind2 = np.where(l1_abs > l2_abs)[0]

        x1s = pts[ind1]
        x2s = pts[ind2]

        # switch over all those in the corners
        try:
            x1_mean = np.mean(x1s, axis=0)
            x2_mean = np.mean(x2s, axis=0)
            side_1 = np.sign(x1_mean @ n2 - 1)
            side_2 = np.sign(x2_mean @ n - 1)

            switch_1 = np.where((l2[ind1]*side_1 < 0) & (l1[ind1]*side_2 > 0))[0]
            switch_2 = np.where((l1[ind2]*side_2 < 0) & (l2[ind2]*side_1 > 0))[0]

            change_1 = ind1[switch_1]
            change_2 = ind2[switch_2]

            ind1 = np.delete(ind1, switch_1)
            ind2 = np.delete(ind2, switch_2)
            ind1 = np.append(ind1, change_2)
            ind2 = np.append(ind2, change_1)
        except Warning:
            pass

        x1s = pts[ind1]
        x2s = pts[ind2]

        n_old = n

        try:
            n, inliers1in = fit_ransac(x1s)
            inliers1 = x1s[inliers1in]
            if len(x2s) < 3:
                x2s = np.append(x2s, np.delete(x1s, inliers1in, 0), axis=0)
        except ValueError:
            pass

        alpha_old = alpha
        alpha, inliers2in = ransac1d(x2s @ R @ n/np.linalg.norm(n), 3, 50)
        alpha *= np.linalg.norm(n)
        inliers2 = x2s[inliers2in]
        # error would have been raised above if this is true so new n has not
        # been calculated
        if len(x1s) < 3:
            x1s = np.append(x1s, np.delete(x2s, inliers2in, 0), axis=0)
            if len(x1s) >= 2:
                n, inliers1 = fit_ransac(x1s)

        if abs(alpha_old-alpha)/alpha_old < tol and  \
                abs(np.linalg.norm(n-n_old))/np.linalg.norm(n_old) < tol:
            break
        n2 = (R @ n)/alpha

    print("Original inliers 2 length {}".format(len(inliers2)))
    c1 = inliers1 @ n - 1
    c2 = (inliers2 @ R @ n)/alpha - 1
    cost = (c1.T @ c1)/len(inliers1) + (c2.T @ c2)/len(inliers2)

    # n_new, i = fit_ransac(x2s, ransac)
    # alpha_new, inliers2in = ransac1d(x1s @ R @ n_new/ np.linalg.norm(n_new), 3, 50)
    # alpha_new *= np.linalg.norm(n_new)
    # j = x1s[inliers2in]
    # c1 = i @ n_new - 1
    # c2 = (j @ R @ n_new)/alpha_new - 1
    # cost2 = (c1.T @ c1)/len(i) + (c2.T @ c2)/len(j)
    # print("Swapped inliers 2 length {}".format(len(j)))

    # since solution is not symmetric, repeat with optimal n rotated and pick
    # best fit
    if top:
        n_new, alpha_new, i, j, cost2 = fit_corner(pts, n=(R @ n)/alpha,
                                                   alpha=np.linalg.norm(n)/alpha)
        if cost2 < cost:
            print("FLIPPED")
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

        print("Return set:")
        print(np.argmin(ang))
        print("Returned inliers 2 length: {}".format(len(return_set[np.argmin(ang)][3])))
        return return_set[np.argmin(ang)]

    return n, alpha, inliers1, inliers2, cost


def ransac1d(pts, min_samp, iter, mads=None):
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
    # min_samp = 500

    # if special.comb(len(pts), min_samp, exact=True) < iter:
    #     combs = np.array(combinations(pts, min_samp))
    #     samples = np.mean(combs)
    # else:
    samples = np.mean(np.random.choice(pts, (min_samp, iter)), axis=0)

    if mads is None:
        # mads = stats.median_absolute_deviation(pts)
        mads = 0.01
    for alpha in samples:
        residuals = np.abs(pts - alpha)
        inliers = np.where(residuals < mads)[0]
        if len(inliers) < min_samp:
            continue
        score = np.linalg.norm(residuals[inliers])**2/len(inliers)
        if score < score_best:
            score_best = score
            inliers_best = inliers

    try:
        return np.mean(pts[inliers_best]), inliers_best
    except UnboundLocalError:
        return np.mean(pts), np.arange(len(pts))

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

def icp_constrained(source, target, theta=0, iter=30, tol=0.05):
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
        R0 = u @ v.T

        R0 = np.array([[R0[0, 0], 0, R0[0, 1], 0],
                       [0,        1, 0,       0],
                       [R0[1, 0], 0, R0[1, 1], 0],
                       [0,        0, 0,       1]])

        trans = mu_t - mu_x @ R0[:3, :3].T
        R0[:3, -1] = trans
        source_c.transform(R0)
        R_old = R0 @ R_old

        if abs(R0[0, 1]) < np.pi*tol:
            break

    # get final cost
    source_c = deepcopy(source)
    source_c.transform(R_old)
    ids = _icp_correspondance(source_c, target_tree)

    # filter outliers
    xs = np.asarray(source_c.points)
    ts = target_points[ids]

    inliers = _icp_get_inliers(xs, ts)
    xs = xs[inliers]
    ts = ts[inliers]

    final_cost = np.sum(np.linalg.norm(xs-ts, axis=0)**2)/len(xs)
    return R_old, final_cost


def icp_constrained_plane(source, target, theta=0, iter=30, tol=0.01):
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

    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()

    R_old = np.array([[np.cos(theta),  0, np.sin(theta), 0],
                      [0,              1,             0, 0],
                      [-np.sin(theta), 0, np.cos(theta), 0],
                      [0,              0,             0, 1]])
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
        R0 = np.identity(4)
        R0[0, 2] = lam[0]
        R0[2, 0] = -lam[0]
        R0[:3, -1] = lam[1:]

        source_c.transform(R0)
        R_old = R0 @ R_old

        if abs(lam[0]) < np.pi*tol and np.linalg.norm(lam[1:]) < tol:
            break

    # get final cost
    source_c = deepcopy(source)
    source_c.transform(R_old)
    ids = _icp_correspondance(source_c, target_tree)

    # filter outliers
    xs = np.asarray(source_c.points)
    ts = target_points[ids]
    ns = target_norms[ids]

    inliers = _icp_get_inliers(xs, ts)
    xs = xs[inliers]
    ts = ts[inliers]
    ns = ns[inliers]

    final_cost = np.sum(np.abs(np.einsum('ij,ij->i', (xs-ts), ns)))/len(xs)
    return R_old, final_cost

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
    r = (lam[0]**2 + lam[1]**2 -4*lam[2])/4
    r = np.sqrt(r)
    return np.array([a,b]), r
