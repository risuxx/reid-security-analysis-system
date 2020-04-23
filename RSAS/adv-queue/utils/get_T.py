import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

def test_T():
    img_path = '../../dataset/bobo/resize/'
    #ori_name = 'IMG_1008.JPG'
    ori_name = '1502_c8l04f0.JPG'
    ori_path = os.path.join(img_path, ori_name)

    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
    pts2 = np.float32([[152,159],[256,152],[150,306],[242,311]])
    #pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])


    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def test_mask():
    img_path = '../../dataset/bobo/resize/'
    #ori_name = 'IMG_1008.JPG'
    ori_name = '1502_c8l04f0.JPG'
    tar_name = '1502_c9l05f0.JPG'
    ori_path = os.path.join(img_path, ori_name)
    tar_path = os.path.join(img_path, tar_name)

    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(tar_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
    pts2 = np.float32([[52,159],[156,152],[50,306],[142,311]])
    #pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])
    M = cv2.getPerspectiveTransform(pts1, pts2)


    mask = np.zeros([550,220,3], dtype=np.uint8)
    mask[140:290,90:185,:] = 1

    noise = np.ones((550,220,3),dtype=np.float32)*100 * mask
    dst = cv2.warpPerspective(noise, M, (220, 550))

    dst2 = img2*(dst<1e-1) + dst
    dst2 = dst2.astype(np.uint8)

    #test for contrib.image.transform
    #with tf.device('/cpu:0'):
    img_tf = tf.Variable(noise)
    M = np.array(np.mat(M).I)
    transform1 = np.reshape(M, 9)[:8]
    #transform1 = [1,0,0,0,1,0,0,0]
    noise2 = tf.contrib.image.transform(img_tf, transform1, 'BILINEAR')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        noise2 = sess.run(noise2)



    #dst = cv2.warpPerspective(img, M, (300,700))

    plt.subplot(141),plt.imshow(img),plt.title('Input')
    plt.subplot(142),plt.imshow(img2),plt.title('Output')
    plt.subplot(143), plt.imshow(noise2), plt.title('Output')
    plt.subplot(144), plt.imshow(dst2), plt.title('Output')
    plt.show()

#当需要增加/修改 某个person的mask时调用
#pid为用户id，pts为一个长度为4的list，注意与trans的顺序不同
def save_mask(pid, pts, dir_path="../../dataset/bobo/mask/"):

    mask = np.zeros([550,220,3], dtype=np.uint8)
    mask[pts[0]:pts[1],pts[2]:pts[3],:] = 1

    np.save(os.path.join(dir_path, str(pid)+'.npy'), mask)

#only excute once
def init():
    dir_path="../../dataset/bobo/mask/"
    pts1 = np.zeros((792,4,2), dtype=np.float32)
    pts2 = np.zeros((792,4,2), dtype=np.float32)
    np.save(os.path.join(dir_path, 'pts1.npy'), pts1)
    np.save(os.path.join(dir_path, 'pts2.npy'), pts2)
    transforms = np.zeros((792,8), dtype=np.float32)
    
    for i in range(792):
        transform = cv2.getPerspectiveTransform(pts1[i], pts2[i])
        transforms[i] = np.reshape(transform,9)[:8]
    
    np.save(os.path.join(dir_path, 'transforms.npy'), transforms)


#当需要增加/修改 某张图片的trans时调用
#img_index:该图片下标；pts1，pts2表示图片相对与a0的映射坐标
def set_pts(img_index, pts1, pts2):
    dir_path = "../../dataset/bobo/mask/"
    pts1_np = np.load(os.path.join(dir_path, "pts1.npy"))
    pts2_np = np.load(os.path.join(dir_path, "pts2.npy"))
    transforms = np.load(os.path.join(dir_path, "transforms.npy"))

    pts1_np[img_index] = pts1
    pts2_np[img_index] = pts2

    trans = cv2.getPerspectiveTransform(pts1, pts2)
    transform = np.array(np.mat(trans).I)
    transforms[img_index] = np.reshape(transform,9)[:8]

    np.save(os.path.join(dir_path, 'pts1.npy'), pts1_np)
    np.save(os.path.join(dir_path, 'pts2.npy'), pts2_np)
    np.save(os.path.join(dir_path, 'transforms.npy'), transforms)

    #oft
    import scipy.misc
    probe_path = '../../dataset' + '/bobo/resize'
    pid = 1502
    mask = np.load(os.path.join(dir_path, str(pid)+'.npy'))
    noise = np.ones((550,220,3),dtype=np.float32)*100 * mask
    dst = cv2.warpPerspective(noise, trans, (220, 550))

    img_path = os.path.join(probe_path,sorted(os.listdir(probe_path))[img_index])
    img = np.array(scipy.misc.imread(img_path),dtype=np.float32)

    dst2 = img*(dst<1e-1) + dst
    dst2 = dst2.astype(np.uint8)
    plt.imshow(dst2)
    plt.show()


test_mask()

#mask[140:290,90:185,:] = 1
#0,4,8,12,18,20,24,28,30,32,40,44,46,50,54,58,62,68,70 ->32
#pts1 = np.float32([[89,139],[189,137],[94,287],[183,290]]) #左上，右上，左下，右下
#pts2 = np.float32([[38,141],[157,142],[41,292],[149,298]])
#set_pts(70, pts1, pts2)

#0 np.float32([[48,127],[177,132],[56,283],[173,284]])
#4 np.float32([[52,130],[189,131],[63,287],[185,288]])
#8 np.float32([[61,144],[151,144],[73,288],[158,290]])
#12 np.float32([[66,132],[168,134],[76,280],[167,281]])
#18 np.float32([[65,130],[165,136],[86,281],[169,284]])
#20 np.float32([[47,142],[140,145],[62,288],[145,291]])
#24 np.float32([[63,142],[191,145],[64,292],[170,298]])
#28 np.float32([[78,158],[182,159],[82,301],[173,305]])
#30 np.float32([[103,156],[199,157],[105,293],[189,299]])
#32 np.float32([[89,139],[189,137],[94,287],[183,290]])
#40 np.float32([[61,140],[171,137],[66,280],[173,288]])
#44 np.float32([[51,141],[139,142],[58,283],[142,289]])
#46 np.float32([[70,142],[170,143],[76,289],[169,289]])
#50 np.float32([[96,155],[182,138],[102,274],[175,283]])
#54 np.float32([[52,152],[139,143],[57,290],[134,295]])
#58 np.float32([[53,157],[157,155],[55,301],[149,307]])
#62 np.float32([[28,146],[130,138],[33,291],[128,296]])
#68 np.float32([[26,152],[119,152],[29,293],[113,303]])
#70 np.float32([[38,141],[157,142],[41,292],[149,298]])