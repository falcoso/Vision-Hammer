"""
Functions for processing a segmenting point clouds for scene reconstruction.

Author O.G.Jones

Functions
---------
register_clouds - Colored pointcloud registration.

region_grow - Segments point cloud using a region growing algorithm.

isolate_model - Isolates a miniature within a point cloud, and returns it
    aligned with the y-axis.

segment - Segments a scene of a warhammer board and classified regions of
    interest.

remove_planes - Removes any planes from a scene.
"""

import open3d as o3d
import numpy as np
import utils

from tqdm import tqdm
from sklearn.cluster import DBSCAN

def register_clouds(target, source,
                    voxel_radius=[0.04, 0.02, 0.01],
                    max_iter=[50, 30, 14],
                    current_transformation=np.identity(4)):
    """
    Colored pointcloud registration.
    This is implementation of following paper
    J. Park, Q.-Y. Zhou, V. Koltun,
    Colored Point Cloud Registration Revisited, ICCV 2017

    Parameters
    ----------
    target: open3d.geometry.PointCloud
        Ground truth point cloud.

    source: open3d.geometry.PointCloud
        Point cloud to which transformation will be applied for aligning.

    voxel_radius: iterable of floats
        List of voxel radii to down sample for each round of iteration

    max_iter: iterable of ints
        List of number of iterations to fit for each voxel radius listed

    current_transformation: 4x4 np.array
        Initial transformation of the source onto the target.

    Returns
    -------
    trans: 4x4 np.array
        Transformation matrix that when applied to source will align it with
        target.
    """

    if len(max_iter) != len(voxel_radius):
        raise TypeError("max_iter and voxel_radius should have the same number of items")

    # calculate normals for point to plane registration
    for cloud in [source, target]:
        if not cloud.has_normals():
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * 2,
                                                     max_nn=30))
    for radius, iter in zip(voxel_radius, max_iter):

        # downsample point cloud to speed up matching
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-7,
                                                    relative_rmse=1e-7,
                                                    max_iteration=iter))

        current_transformation = result_icp.transformation
    return result_icp.transformation

def region_grow(pcd, tol=0.95, find_planes=False):
    """
    Segments point cloud using a region growing algorithm.

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

    plane_normals : np.array of np.array(float) 3-vectors
        Normals for each plane in planes list.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * 2,
                                                                  max_nn=30))

    # create list of indicies to match
    ind = set(range(np.asarray(pcd.points).shape[0]))
    start_size = np.asarray(pcd.points).shape[0]
    planes_list = []
    plane_normals = []
    normals = np.asarray(pcd.normals)
    old_size = start_size

    with tqdm(total=start_size) as t:

        while len(ind) > 0:
            start_point = ind.pop()
            # find it's nearest neighbour
            seeds = {start_point}
            plane = {start_point}
            match_normal = normals[start_point]
            counter = 0  # counter to re-align the vector
            while seeds != set():

                # get next point
                seed = seeds.pop()
                if not find_planes:
                    match_normal = normals[seed]
                # k is number of nearest neighbours, idx is index in list _ is
                #   the distances

                [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[seed], 30)
                # filter any points already classified
                idx = list((set(idx)-plane).intersection(ind))
                kn_normals = normals[idx]
                match = np.abs(kn_normals.dot(match_normal))
                plane_addition = {i for i, j in zip(idx, match) if j > tol}

                # all elements in plane_addition not already in the plane
                seeds.update(plane_addition-plane)
                plane.update(plane_addition)

                # update plane normals if needed
                counter += 1
                counter %= 100
                if counter == 0 and find_planes:
                    plane_points = np.asarray(pcd.select_down_sample(list(plane)).points)
                    pseudoinverse = np.linalg.pinv(plane_points.T)
                    match_normal = pseudoinverse.T.dot(np.ones(plane_points.shape[0]))
                    match_normal /= np.linalg.norm(match_normal)

            # remove all processed points
            ind = ind - plane
            t.update(old_size-len(ind))
            old_size = len(ind)
            planes_list.append(np.array(list(plane), dtype=int))
            if find_planes:
                plane_normals.append(match_normal)

    if find_planes:
        return planes_list, np.array(plane_normals)

    return planes_list


def isolate_model(pcd):
    """
    Isolates a miniature within a point cloud, and returns it aligned with the
    y-axis.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point Cloud of a single miniature on a table.

    Returns
    -------
    cl: open3d.geometry.PointCloud
        Point cloud of just the miniature with its base normal to they y-axis.
    """
    ind = utils.crop_distance(pcd, 0.7)
    cl = pcd.select_down_sample(ind, invert=True)
    pcd = cl.voxel_down_sample(0.01)

    # ragion grow to find planes
    planes_list, norms = region_grow(pcd, tol=0.9, find_planes=True)

    # largest collection will be the table
    plane = planes_list[np.argmax(np.array([len(i) for i in planes_list]))]
    plane = pcd.select_down_sample(np.array(list(plane)))

    # get vectors on plane
    plane_points = np.asarray(plane.points)
    u, s, v = np.linalg.svd(plane_points - np.mean(plane_points, axis=0))
    norm = v[:][2]
    if norm.dot([0, 1, 0]) > 0:
        norm = -norm
    dist = np.mean(plane_points.dot(norm))

    # remove anything below the table
    dot_match = np.abs(np.asarray(cl.points).dot(norm))
    del_ind = np.where(dot_match < 0.97*dist)[0]
    # del_ind [i for i in range(len(dot_match)) if dot_match[i] < 0.97*dist]
    cl = cl.select_down_sample(del_ind)

    # get colours of the table and remove similar colours
    table_colour = np.average(np.asarray(plane.colors), axis=0)
    dot_match = np.abs(np.asarray(cl.colors).dot(table_colour))
    del_ind = [i for i in range(len(dot_match)) if dot_match[i] < 0.7]
    cl = cl.select_down_sample(del_ind)

    cl, ind = utils.clean_cloud(cl, std_ratio=1)

    # Move centre of mass to origin
    centre = cl.get_center()
    cl.translate(-centre)

    # move plane equation - by subtracting plane distance
    dist -= norm.dot(centre)

    # remove final straggles away from centre
    ind = utils.crop_distance(cl, 0.1)
    cl = cl.select_down_sample(ind, invert=True)
    centre = cl.get_center()
    cl.translate(-centre)
    dist = dist - norm.dot(centre)

    # align plane with axis orientation
    axis = np.array([0, -1, 0])  # principle axis of alignment
    cross = -np.cross(axis, norm)
    cos_ang = axis.dot(norm)

    cross_skew = np.array([[0,         -cross[2], cross[1]],
                           [cross[2],  0,         -cross[0]],
                           [-cross[1], cross[0],  0]])

    R = np.identity(3) + cross_skew + np.matmul(cross_skew, cross_skew) * \
        (1-cos_ang)/(np.linalg.norm(cross)**2)

    R = np.array([[R[0][0], R[0][1], R[0][2], 0],
                  [R[1][0], R[1][1], R[1][2], 0],
                  [R[2][0], R[2][1], R[2][2], 0],
                  [0,       0,       0,       1]])

    # add back in plane
    size = 100
    x = np.linspace(-0.1, 0.1, size)
    y = np.linspace(-0.1, 0.1, size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    plane_points = np.array([1, 0, 0])*np.reshape(xv, (len(xv), 1)) + \
        np.array([0, 0, 1])*np.reshape(yv, (len(yv), 1))
    plane.points = o3d.utility.Vector3dVector(plane_points)
    plane.paint_uniform_color([0, 0.5, 0])

    cl = cl.transform(R)
    # place plane at origin
    cl.translate([0, dist, 0])

    return cl

def segment(pcd, *args, **kwargs):
    """
    Segments a scene of a warhammer board and classified regions of interest.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point Cloud to segment
    *args:
        segment_plane *args
    **kwargs:
        segment_plane **kwargs

    Returns
    -------
    labels : numpy.array(int)
        integer label for classifying each point. -1 are outliers, max value
        is the table
    """

    # get table plane
    Normal, plane = pcd.segment_plane(0.005, 20, 100)
    Normal = Normal[:3]/np.linalg.norm(Normal[:3])

    # get the distance of the plane so that everything below can be removed
    dist = np.mean(np.asarray(pcd.points)[plane].dot(Normal))
    n_dot = np.asarray(pcd.points).dot(Normal)-dist

    # classify the scene based on the directions
    plane = np.where(np.abs(n_dot) <= 0.01)[0]  # table itself
    inliers = np.where(n_dot > 0.01)[0]         # everything above the table
    outliers = np.where(n_dot < -0.01)[0]       # anything below the table

    # cluster all inliers
    cluster_model = DBSCAN(eps=0.1,
                           min_samples=50,
                           n_jobs=-1).fit(np.asarray(pcd.points)[inliers])
    cluster_labels = cluster_model.labels_

    # collate all labels
    labels = np.zeros(len(np.asarray(pcd.points)), dtype=int)
    labels[inliers] = cluster_labels
    labels[outliers] = -1
    labels[plane] = np.max(labels)+1
    return labels


def remove_planes(pcd, svd_ratio=20, down_sample=0.01):
    """
    Removes any planes from a scene.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point Cloud to segment
    svd_ratio : float (20)
        The minimum value of s.max()/s.min() for a cluster to be considered a
        plane, where s is the singular values of the cluster's points.

    Returns
    -------
    pcd : open3d.geometry.PointCloud
        Point Cloud with all planes removed
    """
    print("Segmenting Planes...")
    cl = pcd.voxel_down_sample(down_sample)
    planes_list, plane_normals = region_grow(cl, find_planes=True, tol=0.9)
    print("Processing Planes")

    to_remove = []
    filtered_planes = []
    filtered_norms = []
    for i in range(len(planes_list)):
        plane = planes_list[i]
        points = np.asarray(pcd.points)[plane]
        points -= np.mean(points, axis=0)
        s = np.linalg.svd(points, compute_uv=False)

        # planes or lines will have at least 1 small singular value
        if len(s) == 3:
            if s.min() == 0:
                to_remove += list(plane)
                filtered_planes.append(plane)
                filtered_norms.append(plane_normals[i])
            elif s.max()/s.min() > svd_ratio:  # separate line to avoid div0 warnings
                to_remove += list(plane)
                filtered_planes.append(plane)
                filtered_norms.append(plane_normals[i])

    # get largest plane
    filtered_norms = np.array(filtered_norms)
    vec_list = utils.hist_plane_norms(filtered_planes, filtered_norms)

    # find the most common plane normal direction
    floor_vec = vec_list[np.argmax(np.linalg.norm(vec_list, axis=1))]
    floor_vec /= np.linalg.norm(floor_vec)

    # iterate through the normals and find the plane closest to the normal
    max_size = np.max([len(i) for i in filtered_planes])  # size of largest plane
    dist = np.inf
    for plane, normal in zip(filtered_planes, filtered_norms):
        points = np.asarray(cl.points)[plane]
        print(len(plane))
        if floor_vec.dot(normal) > 0.80:
            new_dist = np.mean(points.dot(normal))
            if abs(new_dist) < dist:
                table = plane
                table_norm = normal
                dist = new_dist

    colours = np.random.rand(len(filtered_planes), 3)
    for plane, colour in zip(filtered_planes, colours):
        for point in plane:
            cl.colors[point] = colour

    # remove any other points in line with plane
    n_dot = np.asarray(pcd.points).dot(table_norm)-dist
    to_remove = list(np.where((n_dot > 0.01) | (np.abs(n_dot) < 0.01))[0])
    cl = pcd.select_down_sample(to_remove, invert=True)

    return cl
