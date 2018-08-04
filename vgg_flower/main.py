import os
import pickle
from io import BytesIO
import base64
from flask import Flask, render_template, request
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')   # disappear the bounding of plt painting
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import matplotlib.pyplot as plt

from vgg_flower.model import model_utils
from vgg_flower.tensorflow_vgg import vgg16
from vgg_flower.tensorflow_vgg.utils import load_image


def inference(image):
    """
    model inference by input image
    :param image: input image
    :return: the model output
    """

    # image = load_image(image).reshape(1,224, 224, 3)
    # compute vgg16 features using input image
    vgg = vgg16.Vgg16()               # load VGG16 model
    input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input)

    model_path = os.path.join(model_utils.current_path, 'log')
    # _ = tf.Variable(initial_value='fake_variable')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        features = sess.run(vgg.relu6, feed_dict={input: image})
        output = model_utils.finetuning_model(features, 5)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        return sess.run(tf.nn.softmax(output)).flatten().tolist()


class HandlerLineImage(HandlerBase):
    """
    Replace Matplotlib legend's labels with image
    reference: https://stackoverflow.com/questions/42155119/replace-matplotlib-legends-labels-with-image/42169584
    """

    def __init__(self, image, space=0, offset=0):
        self.space=space
        self.offset=offset
        # self.image_data = plt.imread(path)
        self.image_data = image
        super(HandlerLineImage, self).__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent-30, ydescent-84, width+84, height+84)
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        return [image]


def matplot_image(prediction, image):
    """
    matplotlib draw the prediction to show in flask template
    :param prediction: the model inference output
    :param image: flask index template image
    :return:
    """

    with open(os.path.join(model_utils.current_path, 'labels-list.txt'), 'rb') as f:
        labels = pickle.loads(f.read())
    labels = labels.flatten().tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    rets = ax.bar(range(len(labels)), prediction, 0.5, tick_label=labels)
    plt.legend([rets], [""],
               handler_map={
                   rets: HandlerLineImage(image)},
               fontsize=0, loc="upper center")
    plt.ylim(0, 1.5)

    # show prediction value
    for ret in rets:
        height = ret.get_height()
        ax.text(ret.get_x()+ret.get_width()/2, height, "%.1e" % height, ha='center', va='bottom')


# webapp
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    flask index template
    :return:
    """

    image_data = ""
    if request.method == 'POST':
        image = load_image(request.files['file'])
        prediction = inference(image.reshape(1, 224, 224, 3))
        matplot_image(prediction, image)

        sio = BytesIO()
        plt.savefig(sio, format='jpeg')
        image_data = base64.encodebytes(sio.getvalue()).decode()
        plt.close()

    return render_template('index.html', image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True)
