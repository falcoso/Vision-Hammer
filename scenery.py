"""
Combines all functions into a simple class that does all the processing to
convert a combined warhammer point cloud into a digitally reconstructed scene.

Author O.G.Jones

Functions
---------
open_refs - Opens the reference models for generating a scene.

Classes
-------
Scene - Representation of a simple warhammer board.

Model - Stores all properties of a model recoginised in a warhammer board.
"""

import utils
import correspondance as corr
import copy
import open3d as o3d
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def open_refs(file, mesh):
    """
    Opens the reference models for generating a scene.

    Parameters
    ----------
    file : str
        File title that is the same for all models (but in different folders).
    mesh : bool
        If True, opens the triangle mesh version of the models, otherwise it
        opens the point clouds.

    Returns
    -------
    models : dict
        Description of returned object.

    """
    model_titles = ["Fireblade", "Fire Warrior", "Commander", "Broadside"]
    models = {}
    if mesh:
        io_func = o3d.io.read_triangle_mesh
    else:
        io_func = o3d.io.read_point_cloud
    for model in model_titles:
        tms = io_func("./Point Clouds/Ref - Photogrammetry/{}/{}".format(model, file),
                      print_progress=True)
        models[model] = tms

    if ".obj" in file:
        build_ref = io_func("./Point Clouds/Building Ref.ply",
                            print_progress=True)
        build_ref.compute_triangle_normals()
        build_ref.compute_vertex_normals()
    else:
        build_ref = io_func("./Point Clouds/Building_1000.ply",
                            print_progress=True)
    models["Building"] = build_ref
    return models


class Scene:
    """
    Representation of a simple warhammer board.

    Parameters
    ----------
    cloud : o3d.geometry.PointCloud
        Point cloud representing the scene.
    building : bool (=True)
        True if building is present in the scene. This is used as a size
        reference to more accurately determin scaling.
    align : bool (=True)
        If True, the point cloud will be re-aligned to put the building at the
        centre of the scene, with the table in the x-z plane. If False, this
        alignment must already have been carried out otherwise the
        reconstruction will not be correct.
    seg_scale : bool (=True)
        scale parameter to be passed to segmentation function.
    seg_eps : float (=3)
        eps parameter to be passed to segmentation function for DBSCAN.

    Properties
    ----------
    buidling_height : float
        Height of the building in centimetres.
    scale_factor : float
        Scaling applied to the point clouds to make a distance of 1 in point
        space be equivalent to 1cm.

    Attributes
    ----------
    minis : list (Model)
        List of all the recognised models within the scene.
    building : Model
        Model class representing the building in the scene.
    cloud : o3d.geometry.PointCloud
        Raw point cloud of the scene.

    Methods
    -------
    show_scene(self): Displays all models in the scene.
    show_top_view(self): Displays a birds-eye view of the objects in the scene.
    show_cloud(self, colour_labels=False):
        Shows the original point cloud of the scene.

    """
    buidling_height = 20  # cm
    scale_factor = 93.9919866538415

    def __init__(self, cloud, build=True, align=True, seg_scale=True, seg_eps=3):
        # generate labels
        self.labels, norm = corr.segment(cloud, eps=seg_eps, scale=seg_scale)
        self.cloud = cloud.select_down_sample(np.where(self.labels != -1)[0])

        # save building for quick access
        building_label = -1
        self.building = None
        if build:
            if align:
                self.cloud = corr.building_align(cloud, self.labels, norm)
            max_vol = 0
            for i in range(self.labels.max()):  # note that this will go up to but not include table
                cluster = np.where(self.labels == i)[0]
                cluster = self.cloud.select_down_sample(cluster)
                vol = cluster.get_oriented_bounding_box().volume()
                # building will be largest cluster
                if vol > max_vol:
                    max_vol = vol
                    building = cluster
                    building_label = i

            points = np.asarray(building.points)
            height = points[:, 1].max()-points[:, 1].min()
            Scene.scale_factor = Scene.buidling_height/height  # height of building
            building.scale(Scene.scale_factor, center=False)

            print("Generating Building...")
            self.building = Model(building, "Building")

        else:
            R = utils.align_vectors(norm, np.array([0, 1, 0]))
            self.show_cloud()
            self.cloud.transform(R)
            self.cloud.translate(-self.cloud.get_center())
            table = np.where(self.labels == self.labels.max())[0]
            points = np.asarray(self.cloud.points)
            table_pts = points[table]
            self.cloud.translate(np.array([0, -np.mean(table_pts, axis=0)[1], 0]))
            self.show_cloud()

        self.cloud.scale(Scene.scale_factor, center=False)
        model_labels = [i for i in range(self.labels.max()) if i != building_label]
        targets = [self.cloud.select_down_sample(np.where(self.labels == i)[0])
                   for i in model_labels]

        self.minis = []
        for i in range(len(targets)):
            print("Fitting {} of {} targets...".format(i+1, len(targets)))
            self.minis.append(Model(targets[i]))

        self.cloud = self.cloud.select_down_sample(np.where(self.labels != -1)[0])
        self.labels = self.labels[np.where(self.labels != -1)[0]]
        return

    def show_scene(self):
        """Displays all models in the scene."""
        figures = [i.get_geometry() for i in self.minis]
        if self.building is not None:
            figures.append(self.building.get_geometry())
        o3d.visualization.draw_geometries(figures)
        return

    def show_matches(self):
        figures = [i.get_geometry() for i in self.minis]
        clouds = [i.cluster for i in self.minis]
        if self.building is not None:
            figures.append(self.building.get_geometry())
            clouds.append(self.building.cluster)
        o3d.visualization.draw_geometries(figures+clouds)

    def show_cloud(self, colour_labels=False):
        """
        Shows the original point cloud of the scene.

        Parameters
        ----------
        colour_labels : bool (=False)
            If true, cloud is coloured based on the labels for the cloud.
        """
        cloud = copy.deepcopy(self.cloud)
        if colour_labels:
            cloud = utils.colour_labels(cloud, self.labels)
        o3d.visualization.draw_geometries([cloud])

    def show_top_view(self):
        """Displays a birds-eye view of the objects in the scene."""
        plt.figure()
        for i in self.minis:
            points = i.get_top_points()
            plt.scatter(points[:, 1], points[:, 0])

        if self.building is not None:
            points = self.building.get_top_points()
            plt.scatter(points[:, 1], points[:, 0])
        plt.show()


class Model:
    """
    Stores all properties of a model recoginised in a warhammer board.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud
        Point cloud representing the model.
    ref : str (=None)
        Label identifying what the model is. If no ref is provided, this is
        determined by matching to the ref_dict.
    R : np.array(4x4) (=np.identity(4))
        Transformation to map the reference model to the model's location within
        a scene.

    Properties
    ----------
    inchtocm : float
        conversion of inches to centimetres.
    ref_dict : dict {ref: o3d.geometry.Trianglemesh}
        Dictionary of all the reference models.
    weapons : dict {str: int}
        Dictionary of the ranges of each weapon.
    equipment : dict {ref: list(weapons)}
        Dictionary of the weapons each model has in game.
    movement : dict {ref: int}
        Dictonary of the movement range of each model

    Attributes
    ----------
    cluster : o3d.geometry.PointCloud
        Point cloud representing the model.
    ref : str (=None)
        Label identifying what the model is.
    R : np.array(4x4)
        Transformation to map the reference model to the model's location within
        a scene.

    Methods
    -------
    get_geometry(self): Returns the transformed o3d.geometry of the object.
    get_center(self): Gets the central point of the geometry.
    get_top_points(self): Gets the points in the geometry projected onto the x-z plane.
    """
    inchtocm = 2.54
    ref_dict = {}
    ref_clouds = {}
    ref_circs = {}
    weapons = {"Missile Pod": 36,
               "Pulse Rifle": 30,
               "Markerlight": 36,
               "Fusion Blaster": 18}
    equipment = {"Commander": ["Fusion Blaster"],
                 "Fireblade": ["Markerlight", "Pulse Rifle"],
                 "Broadside": ["Missile Pod"],
                 "Fire Warrior": ["Pulse Rifle"]}
    movement = {"Commander": 20,
                "Fireblade": 7,
                "Broadside": 6,
                "Fire Warrior": 7}

    def __init__(self, cluster, ref=None, R=None):
        if Model.ref_clouds == {}:
            print("Loading Reference Clouds...")
            Model.ref_clouds = open_refs("pcd_1000.ply", False)
            print("Generating Bases...")
            for key, pcd in Model.ref_clouds.items():
                if key == "Building":
                    pass
                points = np.asarray(pcd.points)
                points = points[np.where(points[:,1]<0.5)[0]]
                points = points[:, ::2]
                hull = sp.spatial.ConvexHull(points)
                hull = points[hull.vertices]

                Model.ref_circs[key] = utils.fit_circle(hull)

        self.cluster = cluster
        if ref is not None:
            self.ref = ref
            if R is None: # building will be very roughly aligned already
                R, rmse = utils.icp_constrained_plane(self.cluster, Model.ref_clouds[ref])
            self.R = np.linalg.inv(R)
        else:
            if R is None:
                R = np.identity(4)
            match_best, R = corr.match_model([self.cluster], Model.ref_clouds)
            self.ref = match_best
            self.R = np.linalg.inv(R)
            self.cluster.estimate_normals()
            o3d.visualization.draw_geometries([self.get_geometry().paint_uniform_color([1,0,0]), self.cluster.paint_uniform_color([0,1,0])])

    def get_geometry(self):
        """Returns the transformed o3d.geometry of the object."""
        if self.ref is not None:
            if Model.ref_dict == {}:
                print("Loading References Meshes...")
                Model.ref_dict = open_refs("texturedMesh.obj", True)
            geom = copy.deepcopy(Model.ref_dict[self.ref])
            geom.transform(self.R)
        else:
            geom = self.cluster
        return geom

    def get_center(self):
        """Gets the central point of the geometry."""
        geom = self.get_geometry()
        return geom.get_center()

    def get_top_points(self):
        """Gets the points in the geometry projected onto the x-z plane."""
        geom = self.get_geometry()
        if self.ref is not None:
            points = np.asarray(geom.vertices)
        else:
            points = np.asarray(geom.points)

        points = points[np.where(points[:, 1] > 1)[0]]
        points = points[:, ::2]
        return points
