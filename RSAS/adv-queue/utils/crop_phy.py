import os
import sys
import scipy.misc
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append('../')


def crop(raw_path, crop_path):
    imgs_name = sorted(os.listdir(raw_path))

    c7_index = np.concatenate((np.arange(0,60), np.arange(120,194), np.arange(269, 344)))

    for i in range(len(imgs_name)):
        if i < 269:
            continue
        if i in c7_index:
            camera_id = 7
        else:
            camera_id = 9

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

        new_name = '{:0>4}_c{}s{:0>3}.JPG'.format(person_id, camera_id, i)


        new_path = os.path.join(crop_path, new_name)
        scipy.misc.imsave(new_path, crop)


def crop_simple(raw_path, crop_path):
    imgs_name = sorted(os.listdir(raw_path))

    for i in range(len(imgs_name)):

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

        new_name = '{:0>4}.JPG'.format(i)

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


def crop_mask(logo_path, logo_name, mask_path):
    logo_p = os.path.join(logo_path, logo_name)
    logo = scipy.misc.imread(logo_p)

    plt.figure(figsize=(15, 15), dpi=80)
    plt.imshow(logo)
    pos = np.rint(np.array(plt.ginput(2))).astype(np.int16)
    plt.show()

    crop = logo[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0], :]
    plt.imshow(crop)
    plt.show()

    new_name = 'mask_'+logo_name
    new_path = os.path.join(mask_path, new_name)
    scipy.misc.imsave(new_path, crop.astype(np.uint8))


def get_mask(logo_path, logo_name):
    logo_p = os.path.join(logo_path, logo_name)
    logo = np.array(scipy.misc.imread(logo_p), dtype=np.float32)
    logo = np.concatenate((logo[:,:,0:1], logo[:,:,0:1], logo[:,:,0:1]), axis=2)

    logo = (logo > 100).astype(np.uint8)
    scipy.misc.imsave('mask_qq.png', logo*255)

    logo = scipy.misc.imresize(logo, [420*5, 297*5])
    plt.imshow(logo)
    plt.show()

    np.save('mask_qq.npy', logo)



if __name__ == '__main__':
    raw_path = '../../dataset' + '/wear2_phy3/raw'
    crop_path = '../../dataset' + '/wear2_phy3/crop'
    resize_path = '../../dataset' + '/wear2_phy3/resize'
    #crop_simple(raw_path, crop_path)
    #resize_img(crop_path, resize_path)

    logo_path = '../../dataset' + '/wear2/mask'
    logo_name = 'qq.png'
    mask_path = '../../dataset' + '/wear2/mask'
    #crop_mask(logo_path, logo_name, mask_path)
    logo_name = 'mask_qq.png'
    get_mask(logo_path, logo_name)



