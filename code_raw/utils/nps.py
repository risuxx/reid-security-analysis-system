import numpy as np
#import tensorflow as tf

triplet = np.array([[23,39,6],[21,12,69],[25,62,145],[7,70,84],[16,74,152],
                    [17,126,177],[13,142,61],[17,174,96],[29,145,205],
                    [44,76,99],[99,44,92],[117,21,59],[115,31,123],
                    [120,98,30],[120,162,30],[108,184,45],
                    [170,23,51],[140,24,109],[175,31,136],
                    [151,97,27],[192,177,8],
                    [161,197,15],[218,33,33],[224,15,99],
                    [206,38,141],[195,88,30],[192,187,41],[232,139,40],
                    [202,209,17]],dtype=np.float32)

'''
triplet /=255

for i in range(10):
    c = np.random.randint(0,255,3).astype(np.float32)
    c /= 255

    product = 1.0
    for j in range(0, len(triplet)):
        b = np.sum(np.fabs(c - triplet[j]))
        product *= b

    print(product)

'''













'''
h = 420 * 5
w = 297 * 5
r = np.ones((h, w, 1), dtype=np.float32) * (0)
g = np.ones((h, w, 1), dtype=np.float32) * (0)
b = np.ones((h, w, 1), dtype=np.float32) * (0)
modifier = np.concatenate((r, g, b), axis=2)
modifier = np.random.normal(0,1,(h,w,3))
#modifier = np.ones((h,w,3))


modifier = tf.Variable(modifier, name='modifier', dtype=tf.float32)
noise = (tf.tanh(modifier)+1)*255.0/2

#resize. local mean
filter = tf.fill(value=1.0 / 25, dims=[5, 5, 1, 1])
noise_4D = tf.expand_dims(noise, axis=0)
r = tf.nn.conv2d(noise_4D[:, :, :, 0:1], filter, strides=[1, 5, 5, 1], padding='SAME')
g = tf.nn.conv2d(noise_4D[:, :, :, 1:2], filter, strides=[1, 5, 5, 1], padding='SAME')
b = tf.nn.conv2d(noise_4D[:, :, :, 2:3], filter, strides=[1, 5, 5, 1], padding='SAME')
noise_small = tf.squeeze(tf.concat([r, g, b], axis=3))  # local average scaling

#compute NPS
noise_nps = tf.stack([noise_small] * len(triplet)) / 255.0
triplets = np.array([np.tile(i, (h // 5, w // 5, 1)) for i in triplet]) / 255.0

sub = tf.abs(noise_nps - triplets)

pixel_nps = tf.reduce_sum(sub, axis=3)

prods = tf.reduce_min(pixel_nps, axis=0)

nps = tf.reduce_sum(prods)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    nps_np = sess.run(nps)

    print(nps_np)
'''