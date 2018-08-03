import os
import tensorflow as tf
import numpy as np
import csv
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

from vgg_flower.tensorflow_vgg import vgg16
from vgg_flower.tensorflow_vgg import utils


current_path = os.path.dirname(__file__)
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))


def compute_vgg16_feature():
    """
    compute the datasets(flower photos) feature by vgg16
    :return: features - images features of vgg16 output
             labels - images labels
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


def save_datasets(features, labels):
    """
    save features and labels to disk
    :param features: the features of vgg16 outputs
    :param labels: the labels of images
    :return:
    """

    np.save(os.path.join(current_path, 'features.npy'), features)
    with open(os.path.join(current_path, 'labels.txt'), 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)


def read_datasets(features_file, labels_file):
    """
    read out features, labels from features file and labels file
    :param features_file: features file
    :param labels_file: labels file
    :return: features and labels
    """

    features = np.load(features_file)
    with open(labels_file, 'r') as f:
        labels = [label.strip('\n') for label in f.readlines()]

    return features, labels


def onehot_labels(labels):
    """
    convert label to one-hot format
    :param labels:
    :return: one-hot format labels
    """

    lb = LabelBinarizer()
    lb.fit(labels)
    labels_vecs = lb.transform(labels)

    return labels_vecs


def divide_datasets(features, labels):
    """
    divide datasets to train-sets, validation-sets, and test-sets
    :param features: the features of vgg16 output
    :param labels: the labels of images
    :return: train-sets, validation-sets, and test-sets
    """

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, val_idx = next(ss.split(features, labels))

    half_val_len = int(len(val_idx) / 2)
    val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

    train_x, train_y = features[train_idx], labels[train_idx]
    val_x, val_y = features[val_idx], labels[val_idx]
    test_x, test_y = features[test_idx], labels[test_idx]

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_batches(x, y, n_batch=10):
    """
    split datasets(x, y) to n small batches
    :param x: data of datasets
    :param y: label of datasets
    :param n_batch: number of batch
    :return:
    """

    batch_size = len(x) // n_batch
    for ii in range(0, n_batch*batch_size, batch_size):
        # if this batch isn't the last batch, it should have data of batch_size
        if ii != (n_batch-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size]
        # the rest data, full the batch_size or not, all in the last batch
        else:
            X, Y = x[ii:], y[ii:]

        yield X, Y


def compute_cost(Y, output):
    """
    compute the cost for the model
    :param Y: the datasets labels
    :param output: fine-tuning model output
    :return: cost
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output)
    cost = tf.reduce_mean(cross_entropy)

    return cost


def compute_accuracy(Y, output):
    """
    compute the accuracy for the model
    :param Y: the datasets lables
    :param output: the fine-tuning model output
    :return: accuracy
    """

    prediction = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(correct_prediction, tf.float32)

    return accuracy


def finetuning_model(X, Y):
    """
    fine tuning the vgg16 model for training the flower recognition
    :param X: the datasets input
    :param Y: the dataset label
    :return: the fine tuning model output
    """

    # add densely connected layers
    fc = tf.contrib.layers.full_connected(X, 256)
    output = tf.contrib.layers.full_connected(fc, Y.shape[1], activation_fn=None)

    return output


def main():
    features, labels = compute_vgg16_feature()
    save_datasets(features, labels)
    features, labels = read_datasets(os.path.join(current_path,'features.npy'), os.path.join(current_path, 'labels.txt'))


if __name__ == '__main__':
    main()
