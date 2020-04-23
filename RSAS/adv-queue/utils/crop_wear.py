import os
import sys
import scipy.misc
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append('../')

MAX_PROBE_IMG = 34*2+36

# the step to crop in this way is not neccessary in our exp
def crop(raw_path, crop_path):
    imgs_name = sorted(os.listdir(raw_path))

    for i in range(len(imgs_name)):
        img_path = os.path.join(raw_path, imgs_name[i])
        x = np.array(scipy.misc.imread(img_path), dtype=np.float32)

        camera_id = i//120 + 7
        location_id = i%60//5
        face_id = i%5
        person_id = (i%120)//60+1502

        box_list = [
                    [[480, 150], [400, 155], [370, 145], [340, 125], [390,125], [350,125],
                     [320, 125], [290, 110], [330, 95], [300, 95], [270, 85], [250, 80]],
                    [[460, 140], [420, 130], [350, 120], [300, 100], [465, 150], [410, 130],
                     [360, 120], [310, 100], [470, 150], [400, 130], [350, 120],[300, 110]],
                    [[330, 100],[310,100],[280,90],[250,80],[390,125],[350,125],
                    [320, 125],[290, 110],[480, 150], [420, 155],[380, 140],[330,130]]
                    ]

        s_list = [
            [
            [621, 335], [621, 365], [621, 315], [621, 425], [621, 271],
            [637, 270], [637, 280], [637, 240], [637, 350], [637, 165],
            [641, 196], [641, 196], [641, 185], [641, 250], [641, 145],
            [649, 167], [649, 167], [649, 160], [649, 200], [649, 110],
            [625, 755], [625, 790], [625, 730], [625, 800], [625, 710],
            [640, 646], [640, 660], [640, 630], [640, 670], [640, 610],
            [644, 555], [644, 561], [644, 550], [644, 590], [644, 515],
            [652, 470], [652, 490], [652, 460], [652, 510], [652, 440],
            [630, 1037], [630, 1080], [630, 1010], [630, 1080], [630, 1010],
            [635, 920], [635, 960], [635, 900], [635, 950], [635, 890],
            [645, 819], [645, 850], [645, 800], [645, 850], [645, 790],
            [648, 731], [648, 760], [648, 715], [648, 770], [648, 715]
            ],
            [
            [623, 420], [610, 390], [623, 430], [623, 475], [623, 340],
            [630, 518], [630, 500], [630, 518], [630, 550], [630, 455],
            [648, 570], [648, 560], [648, 584], [648, 615], [648, 520],
            [653, 628], [653, 610], [653, 628], [653, 660], [653, 580],
            [615, 900], [615, 900], [615, 900], [615, 975], [615, 835],
            [630, 915], [630, 915], [630, 910], [630, 975], [630, 845],
            [640, 922], [640, 922], [640, 922], [640, 972], [640, 860],
            [647, 932], [647, 932], [647, 932], [647, 975], [647, 870],
            [610, 1395], [610, 1430], [610, 1385], [610, 1490], [610, 1310],
            [625, 1330], [625, 1350], [625, 1313], [625, 1395], [625, 1260],
            [636, 1275], [636, 1290], [636, 1265], [636, 1320], [636, 1220],
            [644, 1235], [644, 1245], [644, 1220], [644, 1290], [644, 1190],
             ],
            [
            [640,790], [640,720], [640,792], [640,800], [640,730],
            [645, 909], [645, 865], [645, 920], [645, 930], [645, 860],
            [651, 1007], [651, 980], [651, 1020], [651, 1030], [651, 970],
            [654,1100], [654, 1065], [654, 1120], [654, 1130], [654, 1065],
            [632,1038], [632,1000], [632,1060], [632,1080], [632,975],
            [640, 1160], [640,1120], [640,1180], [640, 1200], [640, 1100],
            [635, 1245], [635, 1210], [635, 1265], [635, 1280], [635, 1190],
            [650, 1334], [650, 1310], [650, 1350], [650, 1365], [650, 1275],
            [620, 1425], [620, 1400], [620, 1435], [620, 1470], [620, 1340],
            [625, 1520], [625, 1480], [625,1530], [625, 1580], [625, 1410],
            [630, 1570], [630, 1545], [630, 1590], [630, 1640], [630, 1500],
            [640, 1630], [640, 1610], [640, 1650], [640, 1690], [640, 1560]
            ]
        ]

        #bbox for each camera, location
        index_cam = i // 120
        index_loc = i % 60 // 5
        height, weight = box_list[index_cam][index_loc]

        # startpoint for each camera, location, face
        index_cam = i // 120
        index_loc_face = i % 60
        s = s_list[index_cam][index_loc_face]


        x = x[s[0]:s[0]+height, s[1]:s[1]+weight, :]
        x = x.astype(np.uint8)

        new_name = '{:0>4}_c{}l{:0>2}f{}.JPG'.format(person_id, camera_id,
                                                       location_id, face_id)
        new_path = os.path.join(crop_path, new_name)
        scipy.misc.imsave(new_path, x)

#reszie all the cropped image into (550, 220)
def resize_img(crop_path, resize_path):
    for image_name in sorted(os.listdir(crop_path)):
        image_path = os.path.join(crop_path, image_name)
        # x = np.array(scipy.misc.imresize(scipy.misc.imread(image_path),(550,220)), dtype=np.float32)
        x = cv2.resize(cv2.imread(image_path),(220,550))
        x.astype(dtype=np.uint8)
        image_path = os.path.join(resize_path, image_name)
        # scipy.misc.imsave(image_path, x)
        cv2.imwrite(image_path, x)


def get_pts(resize_path, mask_path):
    imgs_name = sorted(os.listdir(resize_path))

    pts2 = np.zeros((MAX_PROBE_IMG, 4, 2), dtype=np.int64)
    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        # x = np.array(scipy.misc.imread(img_path), dtype=np.uint8)
        x = cv2.imread(img_path)

        plt.figure(figsize=(15,15), dpi=80)
        plt.imshow(x), plt.title(imgs_name[i])
        pos = plt.ginput(4)

        pts2[i] = np.rint(np.array(pos)).astype(np.int64)
        print(pts2[i])  # here to load the pts.npy

        plt.show()

    np.save(os.path.join(mask_path, 'pts2.npy'), pts2)

    # pts1 = np.concatenate((np.tile(pts2[80],(180,1,1)), np.tile(pts2[260],(180,1,1))), axis=0)
    # np.save(os.path.join(mask_path, 'pts1.npy'), pts1)
    set_pts1()

def set_pts1():
    # pts = np.array([[a,b], [c-1, b], [a, d-1], [c-1, d-1]])
    # c+ -> 上面变长  
    # pts = np.array([[0,0], [297*5-1, 0], [0, 420*5-1], [297*5-1, 420*5-1]])
    # pts = np.array([[0,0], [297-1, 0], [0, 420-1], [297-1, 420-1]])
    pts = np.array([[0,0], [220-1, 0], [0, 550-1], [220-1, 550-1]])
    # 左上，右上，左下，右下
    """
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])
    """
    pts1 = np.tile(pts, (MAX_PROBE_IMG, 1, 1))
    np.save(os.path.join(mask_path, 'pts1.npy'), pts1)

def get_tranformations(mask_path):
    pts1 = np.load(os.path.join(mask_path, 'pts1.npy')).astype(np.float32)
    pts2 = np.load(os.path.join(mask_path, 'pts2.npy')).astype(np.float32)
    transforms = np.zeros((MAX_PROBE_IMG, 8), dtype=np.float32)

    for i in range(len(pts2)):
        transform = cv2.getPerspectiveTransform(pts1[i], pts2[i])
        transform = np.array(np.mat(transform).I)
        print(i)
        transforms[i] = np.reshape(transform, 9)[:8]

    np.save(os.path.join(mask_path, 'transforms.npy'), transforms)


def get_mask(pid, pts, mask_path):
    mask = np.zeros([550, 220, 3], dtype=np.uint8)
    mask[pts[0]:pts[1], pts[2]:pts[3], :] = 1
    np.save(os.path.join(mask_path, str(pid) + '.npy'), mask)


def set_transformations(resize_path, mask_path, index):
    pts1 = np.load(os.path.join(mask_path, 'pts1.npy')).astype(np.float32)
    pts2 = np.load(os.path.join(mask_path, 'pts2.npy')).astype(np.float32)

    imgs_name = sorted(os.listdir(resize_path))
    img_path = os.path.join(resize_path, imgs_name[index])


    x = np.array(scipy.misc.imread(img_path), dtype=np.uint8)

    plt.figure(figsize=(15, 15), dpi=80)
    plt.imshow(x), plt.title(imgs_name[index])
    pos = plt.ginput(4)

    print(pts2[index])
    pts2[index] = np.rint(np.array(pos)).astype(np.int64)
    print(pts2[index])

    plt.show()
    pts1 = np.concatenate((np.tile(pts2[80], (180, 1, 1)), np.tile(pts2[260], (180, 1, 1))), axis=0)

    np.save(os.path.join(mask_path, 'pts1.npy'), pts1)
    np.save(os.path.join(mask_path, 'pts2.npy'), pts2)


def test(resize_path, mask_path, test_path):
    imgs_name = sorted(os.listdir(resize_path))

    # pid = 1502
    pid = 2001

    mask = np.load(os.path.join(mask_path, str(pid)+'.npy'))
    noise = mask.astype(np.float32)

    trans = np.load(os.path.join(mask_path, 'transforms.npy'))
    trans = np.concatenate((trans, np.array([1]*MAX_PROBE_IMG).reshape((MAX_PROBE_IMG,1))), axis=1).reshape(MAX_PROBE_IMG,3,3)

    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        # x = np.array(scipy.misc.imread(img_path), dtype=np.float32)
        x = np.array(cv2.imread(img_path),dtype=np.float32)

        trans[i] = np.array(np.mat(trans[i]).I)
        dst = cv2.warpPerspective(noise, trans[i], (220, 550))

        dst2 = x * (dst < 1e-1) + dst
        dst2.astype(np.uint8)
        # scipy.misc.imsave(os.path.join(test_path,imgs_name[i]), dst2)
        cv2.imwrite(os.path.join(test_path,imgs_name[i]), dst2)


def test2(resize_path, test_path):
    imgs_name = sorted(os.listdir(resize_path))[:MAX_PROBE_IMG]
    # noise = np.ones((420*5, 297*5, 3), dtype=np.float32)*60
    noise = np.ones((550, 220, 3), dtype=np.float32)*60

    trans = np.load(os.path.join(mask_path, 'transforms.npy'))
    trans = np.concatenate((trans, np.array([1]*MAX_PROBE_IMG).reshape((MAX_PROBE_IMG,1))), axis=1).reshape(MAX_PROBE_IMG,3,3)

    for i in range(len(imgs_name)):
        img_path = os.path.join(resize_path, imgs_name[i])
        # x = np.array(scipy.misc.imread(img_path), dtype=np.float32)
        x = np.array(cv2.imread(img_path), dtype=np.float32)

        trans[i] = np.array(np.mat(trans[i]).I)
        dst = cv2.warpPerspective(noise, trans[i], (220, 550))

        dst2 = x * (dst < 1e-1) + dst
        dst2.astype(np.uint8)
        # scipy.misc.imsave(os.path.join(test_path,imgs_name[i]), dst2)
        cv2.imwrite(os.path.join(test_path,imgs_name[i]), dst2)

if __name__ == '__main__':
    """
    raw_path = '../../dataset' + '/wear/raw'
    crop_path = '../../dataset' + '/wear/crop'
    resize_path = '../../dataset' + '/wear/resize'
    mask_path = '../../dataset' + '/wear/mask'
    test_path = '../../dataset' + '/wear/test_trans'"""
    raw_path = '../../dataset' + '/20200114/raw'
    crop_path = '../../dataset' + '/20200114/crop'
    resize_path = '../../dataset' + '/20200114/resize'
    mask_path = '../../dataset' + '/20200114/mask'
    test_path = '../../dataset' + '/20200114/test_trans'
    #crop(raw_path, crop_path)
    # mask会影响noise的大小
    # get_mask(2001, [64,136,146*2,226*2], mask_path)
    resize_img(crop_path, resize_path)
    get_pts(resize_path, mask_path)
    # get_tranformations(mask_path)

    # get_mask(1502, [131,277,62,186], mask_path)
    #get_mask(1503, [124,270,55,181], mask_path)
    # test(resize_path, mask_path, test_path)

    ################################# up is the sequentail process
    ################################# down is the reset of certain pts
    #set_transformations(resize_path, mask_path, 51)
    #get_tranformations(mask_path)
    #test(resize_path, mask_path, test_path)

    ##########################we change the a0, big size image
    # set_pts1()
    get_tranformations(mask_path)
    ################################## down is the reset of certain pts
    # set_transformations(resize_path, mask_path, 8) #only change pts, no trans
    # set_pts1()
    # get_tranformations(mask_path)
    # test2(resize_path, test_path)
