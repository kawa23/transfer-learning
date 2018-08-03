import os
from flask import Flask, render_template
import tensorflow as tf

from vgg_flower.model import model_utils
from vgg_flower.tensorflow_vgg import vgg16


def inference(image):
    """
    model inference by input image
    :param image: input image
    :return: the model output
    """

    # compute vgg16 features using input image
    image.reshape((1, 224, 224, 3))    # vgg16 input image must be the shape(224, 224, 3)
    vgg = vgg16.Vgg16()               # load VGG16 model
    vgg.build(image)

    model_path = os.path.join(model_utils.current_path, 'log')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        features = sess.run(vgg.relu6)
        output = model_utils.finetuning_model(features, 5)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return None

        return sess.run(tf.nn.softmax(output))


# webapp
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():

    return render_template('index.html')


if __name__ == '__main__':
    # app.run(debug=True)
    inference(None)