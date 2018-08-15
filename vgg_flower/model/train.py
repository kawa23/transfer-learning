import os
import threading
import datetime

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

    train_x, train_y, val_x, val_y, test_x, test_y = model_utils.divide_datasets(features, labels)

    X = tf.placeholder(tf.float32, [None, features.shape[1]])
    Y = tf.placeholder(tf.int64, [None, train_y.shape[1]])

    output = model_utils.finetuning_model(X, int(Y.shape[1]))

    cost = model_utils.compute_cost(Y, output)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    accuracy = model_utils.compute_cost(Y, output)
    global_step = 0
    epochs = 1000

    # train model
    saver = tf.train.Saver()
    model_path = os.path.join(current_path, 'log')
    initial_epoch = 0
    time_begin = datetime.datetime.now()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # start training begin latest checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_epoch = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

        for epoch in range(initial_epoch, epochs):
            for batch_xs, batch_ys in model_utils.get_batches(train_x, train_y):
                global_step += 1
                train_feed = {X: batch_xs, Y: batch_ys}
                _, train_cost = sess.run([optimizer, cost], feed_dict=train_feed)
                print("epoch: %3d/%3d, global step: %3d, Training cost: %.7f" %
                      ((epoch+1), epochs, global_step, train_cost))

            # save the trained model each epoch
            saver.save(sess, os.path.join(model_path, 'my-model'), global_step=epoch+1)

            # compute validation accuracy each epoch
            val_feed = {X: val_x, Y: val_y}
            val_acc = sess.run(accuracy, feed_dict=val_feed)
            print('----------------------------------------------------------')
            print("epoch: %3d/%3d, Validation accuracy: %.4f" %
                  ((epoch+1), epochs, val_acc))
            print('----------------------------------------------------------')
    time_end = datetime.datetime.now()
    print('%d epochs training model finished: %.0fs' % (epochs, (time_end-time_begin).total_seconds()))

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        test_feed = {X: test_x, Y: test_y}
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("\ntest accuracy: %.4f" % test_acc)


if __name__ == '__main__':
    main()
