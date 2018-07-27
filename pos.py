#!/usr/bin/env python

import numpy as np
from opensfm import types
from opensfm import dataset
from opensfm import matching
from opensfm.reconstruction import resect
from opensfm.reconstruction import run_absolute_pose_ransac
from opensfm.reconstruction import bundle_single_view
from opensfm.reconstruction import get_image_metadata

DATASET = "data/bat0"


#
# copy-pasted and modified from
# create_tracks.Command.load_features()
#
def load_feature_descriptions(data):
    features = {}
    for im in data.images():
        _, f, _ = data.load_features(im)
        features[im] = f
    return features

def shot_pos(shot_id):
    if shot_id not in reconstruction.shots:
        #print("%s not listed in recostruction, skipping" % shot_id)
        return

    del reconstruction.shots[shot_id]

    ok, _ = resect(data, graph, reconstruction, shot_id)
    if not ok:
        print("resect failed")
        return

    s = reconstruction.get_shot(shot_id)
    print("%s: trans %s rot %s" % (shot_id, s.pose.translation, s.pose.rotation))


def my_resect(bs, Xs, data, camera, graph, reconstruction, shot_id):
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    bs = np.array(bs)
    Xs = np.array(Xs)
    if len(bs) < 5:
        return False, {'num_common_points': len(bs)}

    threshold = data.config['resection_threshold']
    T = run_absolute_pose_ransac(
        bs, Xs, "KNEIP", 1 - np.cos(threshold), 1000, 0.999)

    R = T[:, :3]
    t = T[:, 3]

    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = int(sum(inliers))

    print("{} resection inliers: {} / {}".format(
        shot_id, ninliers, len(bs)))
    report = {
        'num_common_points': len(bs),
        'num_inliers': ninliers,
    }
    if ninliers >= data.config['resection_min_inliers']:
        R = T[:, :3].T
        t = -R.dot(T[:, 3])
        shot = types.Shot()
        shot.id = shot_id
        shot.camera = camera
        shot.pose = types.Pose()
        shot.pose.set_rotation_matrix(R)
        shot.pose.translation = t
        shot.metadata = get_image_metadata(data, shot_id)
        reconstruction.add_shot(shot)
        bundle_single_view(graph, reconstruction, shot_id, data.config)
        print "ok"
        print "rotation", shot.pose.rotation
        print "translation", shot.pose.translation
        return True, report
    else:
        print "fail"
        return False, report



def calculate_img_pos(shot_id):
    data = dataset.DataSet(DATASET)
    features = load_feature_descriptions(data)
    graph = data.load_tracks_graph()
#    tracks, images = matching.tracks_and_images(graph)
    reconstruction = data.load_reconstruction()[0]

    img_points, img_features, _ = data.load_features(shot_id)

    # use img_feature to get feature length and datatype
    # of an array for storing feature description
    cloud_features = np.empty((0, img_features.shape[1]),
                              dtype=img_features.dtype)
    cloud_coordinates = []
    # test dumping features
    for point_id, point in reconstruction.points.iteritems():
        cloud_features = np.vstack(
            (cloud_features, features[point.feature.img][point.feature.idx])
        )
        cloud_coordinates.append(point.coordinates)


    #print img_features.shape, cloud_features.shape

    matches = matching.match_symmetric(img_features, None,
                                       cloud_features, None,
                                       data.config)

    exif = data.load_exif(shot_id)
    camera = reconstruction.cameras[exif['camera']]
    bs = []
    Xs = []

    for match in matches:
        b = camera.pixel_bearing(img_points[match[0], :2])
        bs.append(b)
        Xs.append(cloud_coordinates[match[1]])
        #print b, img_points[match[0], :2], "-->", cloud_coordinates[match[1]]

    my_resect(bs, Xs, data, camera, graph, reconstruction, shot_id)
#    print Xs

#    print camera



for i in xrange(1, 11):
    calculate_img_pos("%02d.jpg" % i)
    # debug, only do one image
    #break

#calculate_img_pos("01.jpg")