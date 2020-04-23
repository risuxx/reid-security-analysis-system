import os
import sys
import scipy.misc
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append('../')


def crop(raw_path, crop_path):
    imgs_name = sorted(os.listdir(raw_path))

    for i in range(len(imgs_name)):

        if i < 60: camera_id = 7
        elif i < 90: camera_id = 8
        elif i < 150: camera_id = 9
        else: break

        location_id = i // 5
        face_id = i % 5
        person_id = 1502

        img_path = os.path.join(raw_path, imgs_name[i])
        x = np.array(scipy.misc.imread(img_path), dtype=np.uint8)

        plt.figure(figsize=(15,15), dpi=80)
        plt.imshow(x), plt.title(imgs_name[i])
        pos = np.rint(np.array(plt.ginput(2))).astype(np.int16)
        print(pos)
        plt.show()

        crop = x[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0], :]
        plt.imshow(crop)
        plt.show()

        new_name = '{:0>4}_c{}l{:0>2}f{}.JPG'.format(person_id, camera_id,
                                                     location_id, face_id)
        new_path = os.path.join(crop_path, new_name)
        scipy.misc.imsave(new_path, crop)

#reszie all the cropped image into (550, 220)
def resize_img(crop_path, resize_path):
    for image_name in sorted(os.listdir(crop_path)):
        image_path = os.path.join(crop_path, image_name)
        x = np.array(scipy.misc.imresize(scipy.misc.imread(image_path),(550,220)), dtype=np.float32)
        x.astype(dtype=np.uint8)
        image_path = os.path.join(resize_path, image_name)
        scipy.misc.imsave(image_path, x)


def get_pts(resize_path, mask_path):
    imgs_name = sorted(os.listdir(resize_path))

    pts2 = np.zeros((150, 4, 2), dtype=np.float32)
    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        x = np.array(scipy.misc.imread(img_path), dtype=np.uint8)

        plt.figure(figsize=(15,15), dpi=80)
        plt.imshow(x), plt.title(imgs_name[i])
        pos = plt.ginput(4)

        pts2[i] = np.array(pos, dtype=np.float32)
        #pts2[i] = np.rint(np.array(pos)).astype(np.int64)
        print(pts2[i])  # here to load the pts.npy

        plt.show()

    np.save(os.path.join(mask_path, 'pts2.npy'), pts2)

    set_pts1()

def set_pts1():
    pts = np.array([[0,0], [297*5-1, 0], [0, 420*5-1], [297*5-1, 420*5-1]])
    pts1 = np.tile(pts, (150, 1, 1))
    np.save(os.path.join(mask_path, 'pts1.npy'), pts1)

def get_tranformations(mask_path):
    pts1 = np.load(os.path.join(mask_path, 'pts1.npy')).astype(np.float32)
    pts2 = np.load(os.path.join(mask_path, 'pts2.npy')).astype(np.float32)
    transforms = np.zeros((150, 8), dtype=np.float32)

    for i in range(len(pts1)):
        trans = cv2.getPerspectiveTransform(pts1[i], pts2[i])
        transform = np.array(np.mat(trans).I)
        transforms[i] = np.reshape(transform, 9)[:8]

    np.save(os.path.join(mask_path, 'transforms.npy'), transforms)


def get_mask(pid, pts, mask_path):
    mask = np.zeros([550, 220, 3], dtype=np.uint8)
    mask[pts[0]:pts[1], pts[2]:pts[3], :] = 1
    np.save(os.path.join(mask_path, str(pid) + '.npy'), mask)


def set_transformations(resize_path, mask_path, index):
    pts2 = np.load(os.path.join(mask_path, 'pts2.npy')).astype(np.float32)

    imgs_name = sorted(os.listdir(resize_path))
    img_path = os.path.join(resize_path, imgs_name[index])

    x = np.array(scipy.misc.imread(img_path), dtype=np.uint8)

    plt.figure(figsize=(15, 15), dpi=80)
    plt.imshow(x), plt.title(imgs_name[index])
    pos = plt.ginput(4)
    print(pts2[index])
    pts2[index] = np.array(pos, dtype=np.float32)
    print(pts2[index])

    plt.show()
    np.save(os.path.join(mask_path, 'pts2.npy'), pts2)

    get_tranformations(mask_path)


def test(resize_path, mask_path, test_path):
    imgs_name = sorted(os.listdir(resize_path))

    pid = 1502

    mask = np.load(os.path.join(mask_path, str(pid)+'.npy'))
    noise = mask.astype(np.float32)

    trans = np.load(os.path.join(mask_path, 'transforms.npy'))
    trans = np.concatenate((trans, np.array([1]*150).reshape((150,1))), axis=1).reshape(150,3,3)

    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        x = np.array(scipy.misc.imread(img_path), dtype=np.float32)

        trans[i] = np.array(np.mat(trans[i]).I)
        dst = cv2.warpPerspective(noise, trans[i], (220, 550))

        dst2 = x * (dst < 1e-1) + dst
        dst2.astype(np.uint8)
        scipy.misc.imsave(os.path.join(test_path,imgs_name[i]), dst2)


def test2(resize_path, test_path):
    imgs_name = sorted(os.listdir(resize_path))[:150]
    noise = np.ones((420*5, 297*5, 3), dtype=np.float32)*60

    trans = np.load(os.path.join(mask_path, 'transforms.npy'))
    trans = np.concatenate((trans, np.array([1]*150).reshape((150,1))), axis=1).reshape(150,3,3)

    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        x = np.array(scipy.misc.imread(img_path), dtype=np.float32)

        trans[i] = np.array(np.mat(trans[i]).I)
        dst = cv2.warpPerspective(noise, trans[i], (220, 550))

        dst2 = x * (dst < 1e-1) + dst
        dst2.astype(np.uint8)
        scipy.misc.imsave(os.path.join(test_path,imgs_name[i]), dst2)

if __name__ == '__main__':
    raw_path = '../../dataset' + '/wear2/raw'
    crop_path = '../../dataset' + '/wear2/crop'
    resize_path = '../../dataset' + '/wear2/resize'
    mask_path = '../../dataset' + '/wear2/mask'
    test_path = '../../dataset' + '/wear2/test_trans'
    #crop(raw_path, crop_path)
    #resize_img(crop_path, resize_path)
    #get_pts(resize_path, mask_path)
    #get_tranformations(mask_path)

    #get_mask(1502, [131,277,62,186], mask_path)
    #get_mask(1503, [124,270,55,181], mask_path)
    #test2(resize_path, test_path)


    ################################## down is the reset of certain pts
    set_transformations(resize_path, mask_path, 32) #only change pts, no trans
    test2(resize_path, test_path)
