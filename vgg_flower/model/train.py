import os
import threading
import tensorflow as tf

from vgg_flower.model import model_utils


current_path = os.path.dirname(__file__)


def main():
    features_file = os.path.join(current_path, 'features.npy')
    labels_file = os.path.join(current_path, 'labels.txt')

    # get the features and labels
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        features, labels = model_utils.compute_vgg16_feature()
        # start a thread to save datasets
        t = threading.Thread(target=model_utils.save_datasets, args=(features, labels,))
        t.start()
    else:
        features, labels = model_utils.read_datasets(features_file, labels_file)

    labels_vecs = model_utils.onehot_labels(labels)

    train_x, train_y, val_x, val_y, test_x, test_y = model_utils.divide_datasets(features, labels)

    X = tf.placeholder(tf.float32, [None, features.shape[1]])
    Y = tf.placeholder(tf.int64, [None, labels_vecs.shape[1]])

    output = model_utils.finetuning_model(X, Y)

    cost = model_utils.compute_cost(Y, output)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    accuracy = model_utils.compute_cost(Y, output)
    global_step = 0
    epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_xs, batch_ys in model_utils.get_batches(train_x, train_y):
                global_step += 1
                train_feed = {X: batch_xs, Y:batch_ys}
                _, train_cost = sess.run([optimizer, cost], feed_dict=train_feed)
                print("Epoch: %3d/%3d, global step: %3d, Training cost: %.5f" %
                      ((epoch+1), epochs, global_step, train_cost))

            # compute validation accuracy each epoch
            val_feed = {X: val_x, Y: val_y}
            val_acc = sess.run(accuracy, feed_dict=val_feed)
            print("Epoch: %3d/%3d, global step: %3d, Validation accuracy: %.4f" %
                  ((epoch+1), epochs, global_step, val_acc))


if __name__ == '__main__':
    main()