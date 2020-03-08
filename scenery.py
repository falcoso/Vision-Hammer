import utils
import correspondance as corr
import copy
import open3d as o3d
import numpy as np


class Scene:
    buidling_height = 20  # cm

    def __init__(self, cloud):
        # generate labels
        labels, norm = corr.segment(cloud)
        self.cloud = cloud

        # align building with origin
        # self.cloud = corr.building_align(cloud, labels, norm)
        max_vol = 0

        # save building for quick access
        for i in range(labels.max()):  # note that this will go up to but not include table
            cluster = np.where(labels == i)[0]
            cluster = self.cloud.select_down_sample(cluster)
            vol = cluster.get_oriented_bounding_box().volume()
            # building will be largest cluster
            if vol > max_vol:
                max_vol = vol
                building = cluster
                building_label = i

        points = np.asarray(building.points)
        height = points[:, 1].max()-points[:, 1].min()
        scale_factor = Scene.buidling_height/height  # height of building
        self.cloud.scale(scale_factor, center=False)
        building.scale(scale_factor, center=False)

        R = np.array([[0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]])

        self.building = Model(building, "Building", R=R)

        model_labels = [i for i in range(labels.max()) if i != building_label]
        targets = [self.cloud.select_down_sample(np.where(labels == i)[0]) for i in model_labels]

        self.minis = []
        for i in range(len(targets)):
            print("Fitting {} of {} targets".format(i+1, len(targets)))
            self.minis.append(Model(targets[i]))

        self.cloud = self.cloud.select_down_sample(np.where(labels != -1)[0])
        return

    def show_scene(self):
        figures = [i.get_geometry() for i in self.minis]
        figures.append(self.building.get_geometry())
        o3d.visualization.draw_geometries(figures)
        return


class Model:
    ref_dict = {}

    def __init__(self, cluster, ref=None, R=np.identity(4)):
        if Model.ref_dict == {}:
            print("Loading References...")
            Model.ref_dict = utils.open_refs()

        self.cluster = cluster
        if ref is not None:
            self.ref = ref
            self.R = R
        else:
            match_best, R = corr.match_model([self.cluster], Model.ref_dict)
            self.ref = match_best
            self.R = np.linalg.inv(R)

    def get_geometry(self):
        geom = copy.deepcopy(Model.ref_dict[self.ref])
        geom.transform(self.R)
        return geom
