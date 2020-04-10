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
import scipy as sp
import utils

from tqdm import tqdm
from copy import deepcopy
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
                match = np.abs(kn_normals @ match_normal)
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
                    match_normal = pseudoinverse.T @ np.ones(plane_points.shape[0])
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

    # TODO: This should be less than, which is why below the y-axis has to be
    # inverted for correct alignment
    if norm[1] > 0:
        norm = -norm
    dist = np.mean(plane_points @ norm)

    # remove anything below the table
    dot_match = np.abs(np.asarray(cl.points) @ norm)
    del_ind = np.where(dot_match < 0.97*dist)[0]
    # del_ind [i for i in range(len(dot_match)) if dot_match[i] < 0.97*dist]
    cl = cl.select_down_sample(del_ind)

    # get colours of the table and remove similar colours
    table_colour = np.average(np.asarray(plane.colors), axis=0)
    dot_match = np.abs(np.asarray(cl.colors) @ table_colour)
    del_ind = [i for i in range(len(dot_match)) if dot_match[i] < 0.7]
    cl = cl.select_down_sample(del_ind)

    cl, ind = utils.clean_cloud(cl, std_ratio=1)

    # Move centre of mass to origin
    centre = cl.get_center()
    cl.translate(-centre)

    # move plane equation - by subtracting plane distance
    dist -= norm @ centre

    # remove final straggles away from centre
    ind = utils.crop_distance(cl, 0.1)
    cl = cl.select_down_sample(ind, invert=True)
    centre = cl.get_center()
    cl.translate(-centre)
    dist = dist - norm @ centre

    # align plane with axis orientation
    axis = np.array([0, -1, 0])  # principle axis of alignment
    R = utils.align_vectors(norm, axis)

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


def orient_refs(pcd):
    # get table plane
    # planes_list, plane_normals = region_grow(pcd, find_planes=True)
    # labels = np.zeros(len(np.asarray(pcd.points)), dtype=int)
    # lab = 0
    # for i in planes_list:
    #     labels[i] = lab
    #     lab += 1
    #
    # sizes = np.array([len(i) for i in planes_list])
    # Normal = -plane_normals[np.argmax(sizes)]
    # plane = planes_list[np.argmax(sizes)]
    Normal, plane = pcd.segment_plane(0.01, 20, 100)
    Normal *= -1
    Normal = Normal[:3]/np.linalg.norm(Normal[:3])

    # get the distance of the plane so that everything below can be removed
    dist = np.mean(np.asarray(pcd.points)[plane] @ Normal)
    n_dot = np.asarray(pcd.points) @ Normal-dist

    # classify the scene based on the directions
    plane = np.where(np.abs(n_dot) <= 0.01)[0]  # table itself

    norm = Normal
    R = utils.align_vectors(norm, np.array([0, 1, 0]))
    pcd.transform(R)
    R2 = np.identity(4)
    R2[:3, -1] = -pcd.get_center()
    R = R2 @ R
    pcd.translate(-pcd.get_center())
    table = plane
    points = np.asarray(pcd.points)
    table_pts = points[table]
    R2 = np.identity(4)
    R2[1, -1] = -np.mean(table_pts, axis=0)[1]
    pcd.translate(np.array([0, -np.mean(table_pts, axis=0)[1], 0]))
    R = R2 @ R
    return R


def segment(pcd, plane_thresh=0.01, eps=3, scale=True):
    """
    Segments a scene of a warhammer board and classified regions of interest.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point Cloud to segment
    plane_thresh : float (=0.01)
        Maximum distance from the table for a point to be considered a part of
        the table plane.
    eps : float (=3)
        eps parameter for DBSCAN clustering.
    scale : bool (=True)
        If true the scene is scaled such that the largest item has a height of
        20. This should be the building that is in the scene.

    Returns
    -------
    labels : numpy.array(int)
        integer label for classifying each point. -1 are outliers, max value
        is the table.

    Normal : numpy.array(3)
        3 dimensional unit vector for the normal of the table.
    """
    # get table plane
    Normal, plane = pcd.segment_plane(plane_thresh, 20, 100)
    Normal = Normal[:3]/np.linalg.norm(Normal[:3])

    # get the distance of the plane so that everything below can be removed
    dist = np.mean(np.asarray(pcd.points)[plane] @ Normal)
    n_dot = np.asarray(pcd.points) @ Normal - dist
    if Normal[1] < 0:
        n_dot *= -1

    # classify the scene based on the directions
    plane = np.where(np.abs(n_dot) <= plane_thresh)[0]  # table itself
    inliers = np.where(n_dot > plane_thresh)[0]         # everything above the table
    outliers = np.where(n_dot < -plane_thresh)[0]       # anything below the table

    # scale scene to make eps parameter more robust
    if scale:
        height = n_dot[inliers].max()
        pcd.scale(20/height, center=False)

    # cluster all inliers
    cluster_model = DBSCAN(eps=eps,
                           min_samples=50,
                           n_jobs=-1).fit(np.asarray(pcd.points)[inliers])
    cluster_labels = cluster_model.labels_

    # collate all labels
    labels = np.zeros(len(np.asarray(pcd.points)), dtype=int)
    labels[inliers] = cluster_labels
    labels[outliers] = -1
    labels[plane] = np.max(labels)+1
    return labels, Normal


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
    sizes = np.array([len(i) for i in filtered_planes])
    vec_list = utils.hist_normals(np.einsum('i,ij->ij', sizes,filtered_norms))

    # find the most common plane normal direction
    floor_vec = vec_list[np.argmax(np.linalg.norm(vec_list, axis=1))]
    floor_vec /= np.linalg.norm(floor_vec)

    # iterate through the normals and find the plane closest to the normal
    dist = np.inf
    for plane, normal in zip(filtered_planes, filtered_norms):
        points = np.asarray(cl.points)[plane]
        if floor_vec @ normal > 0.80:
            new_dist = np.mean(points @ normal)
            if abs(new_dist) < dist:
                table_norm = normal
                dist = new_dist

    colours = np.random.rand(len(filtered_planes), 3)
    for plane, colour in zip(filtered_planes, colours):
        for point in plane:
            cl.colors[point] = colour

    # remove any other points in line with plane
    n_dot = np.asarray(pcd.points) @ table_norm - dist
    to_remove = list(np.where((n_dot > 0.01) | (np.abs(n_dot) < 0.01))[0])
    cl = pcd.select_down_sample(to_remove, invert=True)

    return cl


def building_align(pcd, labels, norm):
    """
    Aligns the corner of the building with the principle axis at the origin.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Scene containing building.
    labels : numpy.array(int)
        integer label for classifying each point. -1 are outliers, max value
        is the table.
    norm : numpy.array(3)
        3 dimensional unit vector for the normal of the table.

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        Aligned point cloud.

    """

    # aign the table with horizontal
    R = utils.align_vectors(norm, np.array([0, 1, 0]))
    pcd.transform(R)
    pcd.translate(-pcd.get_center())
    table = np.where(labels == labels.max())[0]
    points = np.asarray(pcd.points)
    table_pts = points[table]
    pcd.translate(np.array([0, -np.mean(table_pts, axis=0)[1], 0]))

    # find the building cluster
    building = 0
    max_vol = 0
    for i in range(labels.max()):  # note that this will go up to but not include table
        cluster = np.where(labels == i)[0]
        cluster = pcd.select_down_sample(cluster)
        vol = cluster.get_oriented_bounding_box().volume()
        # building will be largest cluster
        if vol > max_vol:
            max_vol = vol
            building = cluster

    # fit building corners
    # Project points onto x-z plane
    points = np.asarray(building.points)
    points = points[:, ::2]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    hull = sp.spatial.ConvexHull(points[:, :2])
    hull_pts = points[hull.vertices]

    # iterate convex hulls until we have at least 50 points to fit
    while len(hull_pts) < 50:
        points = np.delete(points, hull.vertices, axis=0)
        hull = sp.spatial.ConvexHull(points[:, :2])
        hull_pts = np.append(hull_pts, points[hull.vertices], axis=0)
        x = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_pts)).voxel_down_sample(0.01)
        hull_pts = np.asarray(x.points)
    hull_pts = hull_pts[:, :2]

    R = np.array([[0, -1],
                  [1, 0]])
    n, alpha, ind1, ind2, cost = utils.fit_corner(hull_pts)
    n2 = (R @ n)/alpha
    corner = np.sum(np.linalg.inv(np.array([n, n2])), axis=1)
    corner = np.array([corner[0], 0, corner[1]])

    # ensures vectors point out from centroid of corner
    if np.mean(hull_pts[ind1] @ n) < np.mean(hull_pts @ n):
        n *= -1
    n = np.array([n[0], 0, n[1]])
    n /= np.linalg.norm(n)
    R = utils.align_vectors(n, np.array([1, 0, 0]))
    pcd.translate(-corner)
    pcd.transform(R)
    return pcd


def match_model(clusters, model_clouds):
    """
    Finds the best matching models to a set of clusters.

    Parameters
    ----------
    clusters : list(open3d.geometry.PointCloud)
        List of point clouds to which a model is to be matched.
    model_clouds : dict {str: open3d.geometry.PointCloud}
        Dictionary of reference clouds to which the clusters are to be matched
        to.

    Returns
    -------
    results : list [(str, np.array(4x4))]
        List of results mapping each cluster to a reference label and a 4x4
        transformation matrix.
    """
    ref_heights = []
    for j, i in model_clouds.items():
        points = np.asarray(i.points)
        height = points[:, 1].max()-points[:, 1].min()
        ref_heights.append(height)
    ref_heights = np.array(ref_heights)
    ref_vols = np.array([i.get_oriented_bounding_box().volume() for j, i in model_clouds.items()])

    result = []
    for cluster in clusters:
        points = np.asarray(cluster.points)
        height = points[:, 1].max()-points[:, 1].min()
        matches = []
        for (key, cl), ref_height, ref_vol in zip(model_clouds.items(),
                                                  ref_heights,
                                                  ref_vols):
            if height > 0.6*ref_height and 1.2*ref_height > height:
                vol = cluster.get_oriented_bounding_box().volume()
                # if vol < 2*ref_vol and vol > 0.3*ref_vol:
                matches.append(key)

        rmse_best = np.inf
        match_best = None
        R_best = np.identity(4)
        for i in matches:
            R, rmse = match_to_model(cluster, model_clouds[i])
            if rmse < rmse_best and rmse < 1:
                rmse_best = rmse
                R_best = R
                match_best = i

        result.append((match_best, R_best))

    if len(clusters) == 1:
        return result[0]

    return result


def match_to_model(cluster, target):
    """
    Finds the best alignment of the cluster to a possible target.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud
        Point cloud to be matched
    target : o3d.geometry.PointCloud
        Match candidate.

    Returns
    -------
    R : np.array(4x4)
        4x4 transformation matrix.
    final_cost : float
        Cost per point of the given transformation.
    """
    cluster2 = deepcopy(cluster)
    R = np.identity(4)
    r = -cluster2.get_center()
    R[:3, -1] = r

    # move cluster to central origin
    t_points = np.asarray(target.points)
    points = np.asarray(cluster2.points)
    y_shift = points[:, 1].max()-t_points[:, 1].max()
    R[1, -1] = -y_shift*0.5
    cluster2.transform(R)
    cluster2.estimate_normals()

    thetas = np.linspace(0, 2*np.pi, 10)
    rmse = np.inf
    for theta in tqdm(thetas):
        R1, cost = utils.icp_constrained_plane(cluster2, target, theta=theta)
        if cost < rmse:
            rmse = cost
            R_best = R1

    R = R_best @ R
    cluster2 = deepcopy(cluster)
    return R, rmse
