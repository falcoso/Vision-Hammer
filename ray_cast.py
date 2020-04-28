import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt

from copy import deepcopy

def get_slice(mesh, norm, viewpoint=np.zeros(3)):
    """
    Creates a set of lines that is a slice across the give mesh

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh to be slices. Should have backfaces already culled.
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

    d = vertices @ n
    plane_distance = viewpoint @ n

    row = []
    for triangle in triangles:
        if np.sum(d[triangle] < plane_distance) in {1,2}:
            row.append(triangle)

    row = np.array(row)

    pts_ind = np.unique(row.flatten())

    # select all vertices in original mesh to create a slice sub-mesh
    slice = mesh.select_down_sample(pts_ind)
    triangles = np.asarray(slice.triangles)
    vertices = np.asarray(slice.vertices)

    # group all connected triangles
    tri_labels = slice.cluster_connected_triangles()
    tri_labels = np.asarray(tri_labels[0])

    groups = []
    for i in range(tri_labels.max()+1):
        group_triangles = triangles[tri_labels == i]

        # get vertices in each group and then split them into separate meshes
        group_ind = set([k for k in group_triangles.flatten()])
        groups.append(slice.select_down_sample(list(group_ind)))

    # find plane intersection on each line
    line_points = np.array([[0, 0, 0]])
    labels = []
    label = 0
    for group in groups:
        line = []
        vertices = np.asarray(group.vertices)
        triangles = np.asarray(group.triangles)
        d = vertices @ n

        for triangle in triangles:
            d_sub = d[triangle]
            if np.sum(d_sub < 0) == 1:
                root = triangle[d_sub < 0]
                stems = triangle[d_sub > 0]
            elif np.sum(d_sub > 0) == 1:
                root = triangle[d_sub > 0]
                stems = triangle[d_sub < 0]
            else:
                # should not enter here but it currently does?
                continue
                raise RuntimeError("This triangle should not have made it to the slice")

            root = vertices[root].flatten()
            stems = vertices[stems].reshape(2, 3)
            direc = stems - root
            alpha = direc @ n
            alpha = (plane_distance - root @ n) / alpha
            new_pts = np.array([root + alpha[i]*direc[i] for i in range(len(alpha))])
            new_pts = new_pts.reshape(2, 3)

            for i in new_pts:
                line.append(deepcopy(i))

        line = np.array(line)

        # also sorts the lines
        line = np.unique(line, axis=0)
        labels += [label]*len(line)
        label += 1
        line_points = np.append(line_points, line, axis=0)

    line_points = line_points[1:]

    return line_points, np.array(labels, dtype=int)

def filter_occluded(line_points, labels):
    """
    Filters a set of planar points to those that can be seen from the origin.

    Parameters
    ----------
    line_points : np.array(N,3)
        Set of planar points
    labels : np.array(N) (int)
        Labels grouping the points together to form connected lines

    Returns
    -------
    line_points : np.array(N,3)
        Set of planar points that are not occluded by one another
    labels : np.array(N) (int)
        Labels grouping the points together to form connected lines
    angles_mask :

    """
    # go through each line segement starting with closest to produce angle mask
    centres = np.array([np.mean(line_points[labels == i], axis=0) for i in range(labels.max()+1)])
    distances = np.linalg.norm(centres, axis=1)
    order = np.argsort(distances)
    angles = np.arctan2(line_points[:, 0], -line_points[:, 2])
    angles_mask = np.array([])

    for label in order:
        angles_section = angles[labels == label]

        # get areas occluded by mask
        occluded = np.zeros(len(angles_section), dtype=int)
        for mask in angles_mask:
            occluded += (angles_section < mask[1]) & (angles_section > mask[0])

        # find each new section the line might be split into
        i = 0
        sections = []
        sec = []
        for i in range(len(occluded)):
            if occluded[i] == 0:
                sec.append(i)
            else:
                if len(sec) != 0:
                    sections.append(np.array(sec))
                sec = []
        if len(sec) != 0:
            sections.append(np.array(sec))

        indexes = np.where(labels == label)[0]
        for section in sections:
            new_index = indexes[section]
            labels[new_index] = labels.max() + 1

        # delete any point where label has not been updated
        to_remove = np.where(labels == label)[0]
        # print(to_remove)
        line_points = np.delete(line_points, to_remove, axis=0)
        angles = np.delete(angles, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)


        # update angles_mask
        new_mask = np.array([angles_section.min(), angles_section.max()])
        angles_mask = np.append(angles_mask, new_mask)
        angles_mask = angles_mask.reshape(len(angles_mask)//2, 2)
        angles_mask = angles_mask[np.argsort(angles_mask[:,0])]
        for i in range(len(angles_mask)-1):
            max_val = angles_mask[i,1]
            if max_val > angles_mask[i+1,0]:
                angles_mask[i+1,0] = None
                if max_val > angles_mask[i+1,1]:
                    angles_mask[i+1,1] = max_val
                angles_mask[i,1] = None

        angles_mask = angles_mask.flatten()
        angles_mask = angles_mask[~np.isnan(angles_mask)]
        angles_mask = angles_mask.reshape(len(angles_mask)//2, 2)

    # compress labels to remove gaps
    max_label = labels.max()
    i = 0
    while i < max_label:
        if not (labels == i).any():
            labels[labels < i] -= 1
            max_label -= 1

    return line_points, labels, angles_mask



if __name__ == "__main__":
    base = o3d.io.read_triangle_mesh("Broadside_down.ply")
    base.translate(np.array([0, -7, -6]))
    base = base.remove_unreferenced_vertices()
    base.compute_vertex_normals()
    base.compute_triangle_normals()
    triangles = np.asarray(base.triangles)
    vertices = np.asarray(base.vertices)
    normals = np.asarray(base.triangle_normals)

    base.remove_triangles_by_index(np.where(normals[:, 2] < 0)[0])
    base = base.remove_unreferenced_vertices()

    n = np.array([0,1,0])
    line_points, labels = get_slice(base, n)

    for i in range(labels.max()+1):
        plt.plot(line_points[labels == i, 0], line_points[labels == i, 2], color="C0")

    line_points, labels, angles_mask = filter_occluded(line_points, labels)

    for i in range(labels.max()+1):
        plt.plot(line_points[labels == i, 0], line_points[labels == i, 2], color="C1")

    for point in line_points:
        plt.plot([0,point[0]], [0,point[2]], color="C3", lw=1)
    plt.ylim(-10,-0)

    plt.show()
