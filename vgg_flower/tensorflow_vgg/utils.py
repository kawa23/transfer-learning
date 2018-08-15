import os

import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

from vgg_flower.tensorflow_vgg.vgg16 import Vgg16

# synset = [l.strip() for l in open('synset.txt').readlines()]
current_path = os.path.dirname(__file__)


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


def vgg16_pb():
    """
    convert vgg16 model to tensorflow pb format file
    :return:
    """

    model = Vgg16(os.path.join(current_path, "vgg16.npy"))
    rgb_batch = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_rgb")
    model.build(rgb_batch)

    graph = tf.get_default_graph()
    # serialize to disk
    graph_def = graph.as_graph_def()
    with tf.gfile.GFile(os.path.join(current_path, "vgg16.pb"), "wb") as wf:
        wf.write(graph_def.SerializeToString())

    del graph, graph_def


if __name__ == "__main__":
    # test()
    vgg16_pb()
    restore_graph = tf.Graph()
    restore_graph_def = restore_graph.as_graph_def()
    with restore_graph.as_default():
        with tf.gfile.GFile("vgg16.pb", "rb") as rf:
            restore_graph_def.ParseFromString(rf.read())
        tf.import_graph_def(restore_graph_def, name="")

    # get necessary tensors
    restore_rgb_input = restore_graph.get_tensor_by_name("input_rgb:0")
    restore_prob_tensor = restore_graph.get_tensor_by_name("relu6:0")
    print(restore_prob_tensor)