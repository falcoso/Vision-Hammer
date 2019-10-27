import open3d as o3d
import numpy as np


def hist_normals(pcd, bin_size=0.95, line_set=False):
    """
    Returns a list of most common normals ordered by magnitude for a given point
    cloud. All vectors returned are normalised by the largest magnitude.

    Parameters
    ----------
    pcd: open3d.geometry.PointCloud
        Point cloud to be histogrammed.

    bin_size: float (=0.95)
        tolerance of the dot product to accept a match.

    line_set: bool
        Returns open3d.LineSet if True

    Returns
    -------
    hist_vect: np.array/ open3d.LineSet
    """
    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    # create empty list to bin into
    vec_list = []
    while len(normals) > 1:
        # select first normal and get dot product with all other normals
        normal = normals[0]
        dot_prod = np.abs(normals.dot(normal))
        ind = np.array([i for i in range(len(dot_prod)) if dot_prod[i] > bin_size])

        vec_list.append(normal*len(ind))
        # delete all normals that have been matched
        normals = np.delete(normals, ind, axis=0)

    vec_list = np.abs(np.array(vec_list))
    # normalise vectors based on largest magnitude
    mags = np.max(np.linalg.norm(vec_list, axis=0))
    vec_list /= mags

    if not line_set:
        return vec_list

    histogram = o3d.geometry.LineSet()
    histogram.points = o3d.utility.Vector3dVector(np.append(np.array([[0, 0, 0]]),
                                                            vec_list, axis=0))
    lines = np.array([[0, i] for i in range(len(vec_list))])
    histogram.lines = o3d.utility.Vector2iVector(lines)

    return histogram


def register_clouds(target, source):
    """
    Colored pointcloud registration
    This is implementation of following paper
    J. Park, Q.-Y. Zhou, V. Koltun,
    Colored Point Cloud Registration Revisited, ICCV 2017

    Parameters
    ----------
    target: open3d.geometry.PointCloud
        Ground truth point cloud.

    source: open3d.geometry.PointCloud
        Point cloud to which transformation will be applied for aligning.

    Returns
    -------
    trans: 4x4 np.array
        Transformation matrix that when applied to source will align it with
        target.
    """

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        # downsample point cloud to speed up matching
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # calculate normals for point to plane registration
        for cloud in [source_down, target_down]:
            if not cloud.has_normals():
                cloud.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2,
                                                         max_nn=30))

        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        return result_icp.transformation


def clean_cloud(pcd_fp, std_ratio=1, nb_neighbors=20, view=False, pcd_sl=None):
    """Cleans a point supplied point cloud of statistical outliers.

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

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                             std_ratio=1)
    if view:
        display_inlier_outlier(pcd, ind)

    if pcd_sl is not None:
        o3d.io.write_point_cloud(pcd_sl, cl)

    return cl, ind


def display_inlier_outlier(cloud, ind):
    """Displays the point cloud with points of index in ind highlighted red

    Parameters
    ----------
    cloud : open3d.geometry.PointCloud
        Description of parameter `cloud`.
    ind : numpy.array (int)
        Indexes in original cloud of
    """
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return
