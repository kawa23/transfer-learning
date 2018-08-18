## 基于VGG迁移学习的花朵识别
![](https://github.com/kawa23/transfer-learning/blob/master/vgg_flower/vgg_flower.gif)

### environment
- MBP
- PyCharm CE
- Python 3.6.5

### requirements
- pip install -r requirements.txt

### run
- download dataset: http://download.tensorflow.org/example_images/flower_photos.tgz
- download vgg16.npy
- train & test: model/train.py
- compression model： tensorflow quantization
- inference: main.py