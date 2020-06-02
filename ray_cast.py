from copy import deepcopy
import open3d as o3d
import numpy as np
import tqdm

import matplotlib.pyplot as plt


def backface_cull(mesh, viewpoint):
    """
    Removes all triangles facing away from the viewpoint.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh with normals.
    viewpoint : np.array(3)
        Viewpoint from which to determine relative triangle orientation.

    Returns
    -------
    new_mesh : o3d.geometry.TriangleMesh
        New mesh with backface triangles removed.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.triangle_normals)

    centroids = np.array([np.mean(vertices[triangle], axis=0) for triangle in triangles])
    to_remove = np.where(np.einsum('ij,ij->i', centroids-viewpoint, normals) > 0)[0]
    new_mesh = deepcopy(base)
    new_mesh.remove_triangles_by_index(np.array(to_remove, dtype=int))
    new_mesh = new_mesh.remove_unreferenced_vertices()
    return new_mesh


def get_slice(mesh, norm, viewpoint=np.zeros(3)):
    """
    Creates a set of lines that is a slice across the give mesh. If the slicing
    plane does not intersect with the mesh None, None is returned.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh to be sliced. Should have backfaces already culled.
    norm : np.array(3)
        Normal of the plane that is sliced.
    viewpoint : np.array(3)
        Viewpoint where plane originates.

    Returns
    -------
    line_points : np.array(N,3)
        set of points on the slice
    labels : np.array(N) (int)
        labels grouping the points together to form connected lines
    """

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # system basis to determine ordering
    n1 = np.mean(vertices, axis=0) - viewpoint
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(n1, norm)
    n2 /= np.linalg.norm(n2)

    plane_distance = viewpoint @ norm
    d = vertices @ norm - plane_distance

    row = np.sum(d[triangles] < 0, axis=1)
    row = triangles[(row == 1) | (row == 2)]

    # slice doesn't intersect with mesh
    if len(row) == 0:
        return None, None
    pts_ind = np.unique(row.flatten())

    # select all vertices in original mesh to create a slice sub-mesh
    slice = mesh.select_down_sample(pts_ind)
    d = d[pts_ind]
    triangles = np.asarray(slice.triangles)
    vertices = np.asarray(slice.vertices)

    # group all connected triangles
    tri_labels = slice.cluster_connected_triangles()
    tri_labels = np.asarray(tri_labels[0])

    # find plane intersection on each line
    line_points = np.array([[0, 0, 0]])
    labels = []
    comp_labels = False
    for label in range(tri_labels.max()+1):
        line = []
        triangles_group = triangles[tri_labels == label]

        for triangle in triangles_group:
            d_sub = d[triangle]
            if np.sum(d_sub < 0) == 1:
                root = triangle[d_sub < 0]
                stems = triangle[d_sub > 0]
            elif np.sum(d_sub >= 0) == 1:
                root = triangle[d_sub > 0]
                stems = triangle[d_sub < 0]
            else:
                # should only enter here if two adjacent triangles have valid
                # slice, and hence was carried over in down sample
                continue

            root = vertices[root].flatten()
            stems = vertices[stems].reshape(2, 3)
            direc = stems - root
            alpha = direc @ norm
            alpha = (plane_distance - root @ norm) / alpha
            new_pts = np.array([root + alpha[i]*direc[i] for i in range(len(alpha))])
            new_pts = new_pts.reshape(2, 3)

            for i in new_pts:
                line.append(deepcopy(i))

        # section doesn't actually intersect
        if len(line) == 0:
            tri_labels[tri_labels == label] = -1
            comp_labels = True
            continue

        line = np.unique(np.array(line), axis=0)
        # sort along basis
        N = np.array([n1, n2])
        pts = (line - viewpoint) @ N.T
        angles = np.arctan2(pts[:, 0], pts[:, 1])
        line = line[np.argsort(angles)]

        labels += [label]*len(line)
        line_points = np.append(line_points, line, axis=0)

    line_points = line_points[1:]
    labels = np.array(labels, dtype=int)
    if comp_labels:
        max_label = labels.max()
        i = 0
        while i < max_label:
            if not (labels == i).any():
                labels[labels > i] -= 1
                max_label -= 1
            else:
                i += 1

    return line_points, labels


def filter_occluded(line_points, labels, viewpoint, norm):
    """
    Filters a set of planar points to those that can be seen from the origin.

    Parameters
    ----------
    line_points : np.array(N,3)
        Set of planar points
    labels : np.array(N) (int)
        Labels grouping the points together to form connected lines
    viewpoint : np.array(3)
        Viewpoint where plane originates.

    Returns
    -------
    line_points : np.array(N,3)
        Set of planar points that are not occluded by one another
    labels : np.array(N) (int)
        Labels grouping the points together to form connected lines
    angles_mask :

    """
    def find_clipped_point(pts, angles_mask):
        angles = np.arctan2(pts[:, 0], pts[:, 1])
        # find intersecting mask
        for mask in angles_mask:
            occluded = (angles < mask[1]) & (angles > mask[0])
            if occluded[0] ^ occluded[1]:
                break

        # get the specific occlusion angle
        angle = mask[occluded[0]]
        n = np.array([np.sin(angle), np.cos(angle)])

        d = pts[1] - pts[0]
        d = d[::-1]
        d[0] *= -1

        alpha = pts[0] @ d / (n @ d)

        return alpha*n

    line_points = deepcopy(line_points)
    # get basis vectors for plane.
    n1 = np.mean(line_points, axis=0) - viewpoint
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(n1, norm)
    n2 /= np.linalg.norm(n2)

    # transform points into basis
    N = np.array([n1, n2])
    pts = (line_points - viewpoint) @ N.T

    # go through each line segement starting with closest to produce angle mask
    centres = np.array([np.mean(pts[labels == i], axis=0) for i in range(labels.max()+1)])
    distances = np.linalg.norm(centres, axis=1)
    order = np.argsort(distances)
    angles = np.arctan2(pts[:, 0], pts[:, 1])
    angles_mask = np.array([])

    for label in order:
        angles_section = angles[labels == label]

        # get areas occluded by mask
        occluded = np.zeros(len(angles_section), dtype=int)
        for mask in angles_mask:
            occluded += (angles_section < mask[1]) & (angles_section > mask[0])

        # find each new section the line might be split into
        sections = []
        occluded[occluded > 0] = 1
        diff = np.diff(occluded)
        changes = np.where(diff != 0)[0]

        up = 0
        down = 0
        if len(changes) != 0:
            for change in changes:
                if diff[change] == 1:
                    up = change
                    sections.append(np.arange(down, up))
                elif diff[change] == -1:
                    down = change
                    # new_pts =

            if diff[changes][-1] == -1:
                sections.append(np.arange(down, len(occluded)))
        elif occluded[0] == 0:
            sections.append(np.arange(0, len(occluded)))

        # i = 0
        # sections = []
        # sec = []
        # for i in range(len(occluded)):
        #     if occluded[i] == 0:
        #         sec.append(i)
        #     else:
        #         if len(sec) != 0:
        #             # find new clipped point
        #             sections.append(np.array(sec))
        #         sec = []
        # if len(sec) != 0:
        #     sections.append(np.array(sec))

        indexes = np.where(labels == label)[0]
        for section in sections:
            new_index = indexes[section]
            labels[new_index] = labels.max() + 1

        # delete any point where label has not been updated
        to_remove = np.where(labels == label)[0]
        line_points = np.delete(line_points, to_remove, axis=0)
        angles = np.delete(angles, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

        # update angles_mask
        new_mask = np.array([angles_section.min(), angles_section.max()])
        angles_mask = np.append(angles_mask, new_mask)
        angles_mask = angles_mask.reshape(len(angles_mask)//2, 2)
        angles_mask = angles_mask[np.argsort(angles_mask[:, 0])]
        for i in range(len(angles_mask)-1):
            max_val = angles_mask[i, 1]
            if max_val > angles_mask[i+1, 0]:
                angles_mask[i+1, 0] = None
                if max_val > angles_mask[i+1, 1]:
                    angles_mask[i+1, 1] = max_val
                angles_mask[i, 1] = None

        angles_mask = angles_mask.flatten()
        angles_mask = angles_mask[~np.isnan(angles_mask)]
        angles_mask = angles_mask.reshape(len(angles_mask)//2, 2)

    # compress labels to remove gaps
    max_label = labels.max()
    i = 0
    while i < max_label:
        if not (labels == i).any():
            labels[labels > i] -= 1
            max_label -= 1
        else:
            i += 1

    return line_points, labels, angles_mask


if __name__ == "__main__":
    base = o3d.io.read_triangle_mesh("Broadside_down.ply")
    # base = o3d.io.read_triangle_mesh("./Point Clouds/Ref - Photogrammetry/Broadside/texturedMesh.obj")
    base = base.remove_unreferenced_vertices()
    base.translate([-7,-5,-10])
    base.compute_vertex_normals()
    base.compute_triangle_normals()
    triangles = np.asarray(base.triangles)
    vertices = np.asarray(base.vertices)
    normals = np.asarray(base.triangle_normals)

    # viewpoint = np.array([7, 5, 10])
    viewpoint = np.array([0, 0, 0])
    base = backface_cull(base, viewpoint)
    max_pt = vertices[np.argmax(vertices[:, 1])]-viewpoint
    min_pt = vertices[np.argmin(vertices[:, 1])]-viewpoint

    theta = np.linspace(np.arctan2(min_pt[1], np.linalg.norm(min_pt[::2])), np.arctan2(max_pt[1], np.linalg.norm(max_pt[::2])), 300)

    n = np.zeros((len(theta), 3))
    n[:, 2] = -np.sin(theta)
    n[:, 1] = np.cos(theta)
    # n = [n[126, :]]
    lines = []
    for normal in tqdm.tqdm(n):
        line_points, labels = get_slice(base, normal, viewpoint)
        if line_points is None:
            continue

        # for i in range(labels.max()+1):
        #     plt.plot(line_points[labels == i, 0], line_points[labels == i, 2], color="C0")

        line_points, labels, angles_mask = filter_occluded(line_points, labels, viewpoint, normal)

        # for i in range(labels.max()+1):
        #     plt.plot(line_points[labels == i, 0], line_points[labels == i, 2], color="C1")
        #
        # for point in line_points:
        #     plt.plot([viewpoint[0],point[0]], [viewpoint[2],point[2]], color="C3", lw=1)
        # plt.show()

        for i in range(labels.max()+1):
            vector = o3d.utility.Vector3dVector(line_points[labels == i])
            size = np.sum(labels == i)
            if size == 1:
                continue
            pointers = np.append(np.arange(0, size-1).reshape(1, size-1),
                                 np.arange(1, size).reshape(1, size-1), axis=0).T
            lines.append(o3d.geometry.LineSet(vector, o3d.utility.Vector2iVector(pointers)))

    n = np.zeros((len(theta), 3))
    n[:, 2] = -np.sin(theta)
    n[:, 0] = np.cos(theta)
    # n = n[95:100]
    # n = [np.array([1,0,0])]
    for normal in tqdm.tqdm(n):
        line_points, labels = get_slice(base, normal, viewpoint)
        if line_points is None:
            continue

        # for i in range(labels.max()+1):
        #     plt.plot(line_points[labels == i, 1], line_points[labels == i, 2], color="C0")

        line_points, labels, angles_mask = filter_occluded(line_points, labels, viewpoint, normal)

        # for i in range(labels.max()+1):
        #     plt.plot(line_points[labels == i, 1], line_points[labels == i, 2], color="C1")
        #
        # for point in line_points:
        #     plt.plot([viewpoint[1],point[1]], [viewpoint[2],point[2]], color="C3", lw=1)
        # plt.show()

        for i in range(labels.max()+1):
            vector = o3d.utility.Vector3dVector(line_points[labels == i])
            size = np.sum(labels == i)
            if size == 1:
                continue
            pointers = np.append(np.arange(0, size-1).reshape(1, size-1),
                                 np.arange(1, size).reshape(1, size-1), axis=0).T
            lines.append(o3d.geometry.LineSet(vector, o3d.utility.Vector2iVector(pointers)))

    o3d.visualization.draw_geometries(lines)
