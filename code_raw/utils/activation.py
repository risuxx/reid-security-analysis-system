import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
from keras.layers import Input
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import scipy.misc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load(dir_path, img_names):
    #image, infos = load_raw(dir_path, img_names, index)
    img_path = os.path.join(dir_path, img_names)
    img = scipy.misc.imread(img_path)
    img = np.array([img], dtype=np.float32)

    img1_tf = tf.placeholder(shape=(None, 550, 220, 3), dtype='float32')
    img2_tf = tf.image.resize_images(img1_tf, [224, 224])
    img3_tf = preprocess_input(img2_tf)

    with tf.Session() as sess:
        image = sess.run(img3_tf, feed_dict={img1_tf:img})
    return image

def load_small(dir_path, img_names):
    #image, infos = load_raw(dir_path, img_names, index)
    img_path = os.path.join(dir_path, img_names)
    img = scipy.misc.imread(img_path)
    img = np.array([img], dtype=np.float32)

    img1_tf = tf.placeholder(shape=(None, 128, 64, 3), dtype='float32')
    img2_tf = tf.image.resize_images(img1_tf, [224, 224])
    img3_tf = preprocess_input(img2_tf)

    with tf.Session() as sess:
        image = sess.run(img3_tf, feed_dict={img1_tf:img})
    return image


def activate_map(img):
    img = np.sum(img[0], axis=2)/np.shape(img)[3]

    minx = img.min()
    maxx = img.max()
    img = (img-minx)/(maxx-minx)

    img = (img*255).astype(np.uint8)

    plt.matshow(img, cmap='jet')
    plt.show()

if __name__ == '__main__':
    net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')
    model = Model(inputs=[net.layers[0].input], outputs=[net.layers[11].output])

    dir_path = './'
    img_names = '1502_c7l00f2.png' # adv
    #img_names ='1502_c7l00f0.JPG'  # benign
    #img_names = '0020.JPG'
    #img_names = '1491_c5s3_073812_01.jpg'
    img_names = '1502_c7l02f3.JPG'
    #img_names = '1502_c7l01f3.png'

    img = load(dir_path, img_names)
    #img = load_small(dir_path, img_names)
    preds = model.predict(img)
    activate_map(preds)