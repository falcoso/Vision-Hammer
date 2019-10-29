import open3d as o3d
import numpy as np
import random


def hist_normals(pcd, bin_size=0.95):
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
    # estimate normals
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    # create empty list to bin into
    vec_list = []
    while len(normals) > 1:
        # select first normal and get dot product with all other normals
        normal = normals[0]
        normals = normals[1:-1]
        dot_prod = np.abs(normals.dot(normal))
        ind = np.array([i for i in range(len(dot_prod)) if dot_prod[i] > bin_size])

        vec_list.append(normal*len(ind))
        # delete all normals that have been matched
        normals = np.delete(normals, ind, axis=0)

    vec_list = np.abs(np.array(vec_list))
    # normalise vectors based on largest magnitude
    mags = np.max(np.linalg.norm(vec_list, axis=0))
    vec_list /= mags

    return vec_list


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
    # calculate normals for point to plane registration
    for cloud in [source, target]:
        if not cloud.has_normals():
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * 2,
                                                     max_nn=30))
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        # downsample point cloud to speed up matching
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))

        current_transformation = result_icp.transformation
    return result_icp.transformation


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


def region_grow(pcd, tol=0.95, find_planes=False):
    """
    Segments point cloud using a region growing algorithm

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point Cloud to segment
    tol : float (0.95)
        Value between 0 and 1 for how sensitive to curvature the region growing
        should be
    find_planes : bool (False)
        If True, the region will segment trying to find planes in the model.

    Returns
    -------
    planes_list : list of np.array(int)
        List of arrays containing the indicies for all points in the segement
        from the point cloud.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * 2,
                                                                  max_nn=30))

    # create list of indicies to match
    ind = list(range(np.asarray(pcd.points).shape[0]))
    planes_list = []

    while len(ind) > 0:
        start_point = ind.pop()
        # find it's nearest neighbour
        # k is number of nearest neighbours, idx is index in list _ is the distances
        seeds = {start_point}
        plane = {start_point}
        normals = np.asarray(pcd.normals)
        match_normal = normals[start_point]
        counter = 0  # counter to re-align the vector
        while seeds != set():

            # get next point
            seed = seeds.pop()
            if not find_planes:
                match_normal = normals[seed]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[seed], 30)
            match = np.abs(normals.dot(match_normal))
            plane_addition = {i for i in idx if match[i] > tol}
            seed_addition = {i for i in plane_addition if i not in plane}
            seeds.update(seed_addition)
            plane.update(plane_addition)

            counter += 1
            counter %= 100
            if counter == 0 and find_planes:
                plane_points = np.asarray(pcd.select_down_sample(np.array(list(plane))).points)
                pseudoinverse = np.linalg.pinv(plane_points.T)
                match_normal = pseudoinverse.T.dot(np.ones(plane_points.shape[0]))
                match_normal /= np.linalg.norm(match_normal)

        planes_list.append(np.array(list(plane), dtype=int))
        ind = [i for i in ind if i not in plane]

    return planes_list


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("./Point Clouds/combined_cloud_clean.ply")
    lines = hist_normals(pcd)
    line_set = create_vector_graph(lines)
    o3d.visualization.draw_geometries([pcd, line_set])
