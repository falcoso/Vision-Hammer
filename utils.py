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
    histogram.points = o3d.utility.Vector3dVector(np.append(np.array([[0,0,0]]), vec_list, axis=0))
    lines = np.array([[0,i] for i in range(len(vec_list))])
    histogram.lines = o3d.utility.Vector2iVector(lines)

    return histogram
