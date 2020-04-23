from __future__ import division, print_function, absolute_import

import os
current_path = os.path.dirname(__file__)

from random import shuffle
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, Lambda, Concatenate
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.models import load_model

from numpy.random import randint, shuffle, choice

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def reid_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = int(line.split('_')[0])
            if lbl != last_label:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl

            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            class_img_labels[str(class_cnt)].append(img[0])

    return class_img_labels


def eucl_dist(inputs):
    x, y = inputs
    # return K.mean(K.square((x - y)), axis=1)
    return K.square((x - y))

def get_model():
    base_model = load_model('market_softmax_pretrain.h5')
    net = Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output],
                name='resnet50')

    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')
    img3 = Input(shape=(224, 224, 3), name='img_3')
    feature1 = Flatten()(net(img1))
    feature2 = Flatten()(net(img2))
    feature3 = Flatten()(net(img3))

    category_predict1 = Dense(751, activation='softmax', name='ctg_out_1')(
        Dropout(0.9)(feature1)
    )
    category_predict2 = Dense(751, activation='softmax', name='ctg_out_2')(
        Dropout(0.9)(feature2)
    )
    category_predict3 = Dense(751, activation='softmax', name='ctg_out_3')(
        Dropout(0.9)(feature3)
    )

    tri_loss = Lambda(triplet_loss, name='ranking')([feature1, feature2, feature3])

    model = Model(inputs=[img1, img2, img3], outputs=[category_predict1, category_predict2,
                                                      category_predict3, tri_loss])

    model.get_layer('ctg_out_1').set_weights(base_model.get_layer('fc8').get_weights())
    model.get_layer('ctg_out_2').set_weights(base_model.get_layer('fc8').get_weights())
    model.get_layer('ctg_out_3').set_weights(base_model.get_layer('fc8').get_weights())
    model.summary()

    #net.summary()
    return net, model

def triplet_hard_loss(inputs):
    x, y, z = inputs
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    z = K.l2_normalize(z, axis=1)


def pair_generator(class_img_labels, batch_size, train=False):
    #p_num = len(class_img_labels)
    p_num = 50

    while True:
        img1_label = randint(p_num, size=batch_size)
        img2_label = img1_label
        binary1_label = (img1_label == img2_label).astype(int)

        while True:
            img3_label = randint(p_num, size=batch_size)
            binary2_label = (img1_label == img3_label).astype(int)
            if np.sum(binary2_label) == 0:
                break

        img1 = list()
        img2 = list()
        img3 = list()

        for i in range(batch_size):
            img1_label_i = len(class_img_labels[str(img1_label[i])])
            img1.append(class_img_labels[str(img1_label[i])][choice(img1_label_i)])

            img2_label_i = len(class_img_labels[str(img2_label[i])])
            img2.append(class_img_labels[str(img2_label[i])][choice(img2_label_i)])

            img3_label_i = len(class_img_labels[str(img3_label[i])])
            img3.append(class_img_labels[str(img3_label[i])][choice(img3_label_i)])

        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)

        triplet_label = np.ones(batch_size)
        img1_label = to_categorical(img1_label)
        img2_label = to_categorical(img2_label)
        img3_label = to_categorical(img3_label)

        yield [img1, img2, img3], triplet_label, #[img1_label, img2_label, img3_label, triplet_label]


def triplet_loss(inputs):
    x, y, z = inputs
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    z = K.l2_normalize(z, axis=1)

    dist_p = K.sum(K.square(x-y), axis=1, keepdims=True)
    dist_n = K.sum(K.square(x-z), axis=1, keepdims=True)

    dist_p = K.sqrt(dist_p)
    dist_n = K.sqrt(dist_n)

    loss = K.maximum(0.0, dist_p - dist_n + 10)
    return loss


def softmax_model_pretrain(train_list, train_dir):

    base_model = load_model('market_softmax_pretrain.h5')
    net = Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output],
                name='resnet50')

    for layer in net.layers:
        layer.trainable = True

    img1_tf = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    img2_tf = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    img3_tf = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)

    feature1 = tf.squeeze(net(img1_tf), axis=[1, 2])
    feature2 = tf.squeeze(net(img2_tf), axis=[1, 2])
    feature3 = tf.squeeze(net(img3_tf), axis=[1, 2])

    feature1_norm = tf.nn.l2_normalize(feature1, axis=1)
    feature2_norm = tf.nn.l2_normalize(feature2, axis=1)
    feature3_norm = tf.nn.l2_normalize(feature3, axis=1)

    dist_p = tf.reduce_sum(tf.multiply(feature1_norm, feature2_norm), axis=1)  # p for positive
    dist_n = tf.reduce_sum(tf.multiply(feature1_norm, feature3_norm), axis=1)  # n for nagetive

    dist_p = tf.reduce_mean(dist_p)
    dist_n = tf.reduce_mean(dist_n)

    dist_loss = tf.maximum(0.0, dist_n - dist_p + 0.6)

    train_step = tf.train.AdamOptimizer(0.0003).minimize(dist_loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        class_img_labels = reid_data_prepare(train_list, train_dir)
        batch_size = 16
        train_generator = pair_generator(class_img_labels, batch_size, train=False)
        base_model.load_weights('market_softmax_pretrain.h5')

        for i in range(1000):

            data, label = next(train_generator)
            _, distp_np, distn_np, tmp = sess.run([train_step, dist_p, dist_n, dist_loss],\
                                                  feed_dict={img1_tf: data[0], img2_tf: data[1], img3_tf: data[2]})
            print(i, distp_np, distn_np, tmp)

            # names = [v.name for v in tf.trainable_variables()]
            #print(sess.run(names[100:101]))

    net.save('multi-task.h5')


def softmax_pretrain_on_dataset(source, project_path='../', dataset_parent='../../dataset'):
    train_list = project_path + '/dataset/market_train.list'
    train_dir = dataset_parent + '/Market-1501/bounding_box_train'
    softmax_model_pretrain(train_list, train_dir)


if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    sources = ['market']
    for source in sources:
        softmax_pretrain_on_dataset(source)
