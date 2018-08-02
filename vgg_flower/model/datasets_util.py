import os
import tensorflow as tf
import numpy as np
import csv

from vgg_flower.tensorflow_vgg import vgg16
from vgg_flower.tensorflow_vgg import utils


current_path = os.path.dirname(__file__)
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))


def compute_vgg16_feature():
    """
    compute the datasets(flower photos) feature by vgg16
    :return:
    """

    data_dir = os.path.join(parent_path, 'flower_photos')
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(os.path.join(data_dir, each))]

    batch_size = 10
    labels = []
    batch = []
    features = None
    # load VGG16 model
    vgg = vgg16.Vgg16()
    input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input)
    with tf.Session() as sess:
        # compute feature for each type of flowers
        for each in classes:
            print("Starting {} images".format(each))
            class_path = os.path.join(data_dir, each)
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):
                # load image to batch list
                img = utils.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)

                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)
                    feed_dict = {input: images}
                    features_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    # store computed features at codes list
                    if features is None:
                        features = features_batch
                    else:
                        features = np.concatenate((features, features_batch))
                    # clear batch for next batch to compute feature
                    batch = []
                    print('{} images precoessed'.format(ii))

    return features, labels


def main():
    features, labels = compute_vgg16_feature()
    print(features)
    print(labels)


if __name__ == '__main__':
    main()