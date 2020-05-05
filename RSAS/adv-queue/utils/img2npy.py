import numpy as np
import scipy.misc
import os

def mask2npy(adv_id, mask_path, make_name):
    img_name = os.path.join(mask_path, make_name)
    image = scipy.misc.imread(img_name, mode='RGB')
    # resize_image = scipy.misc.imresize(image,size=(550,220))
    # rgb_image = np.repeat(resize_image[:, :, np.newaxis], 3, axis=2)
    print(image.shape)

    mask_file = os.path.join(mask_path, '{}.npy'.format(adv_id))
    np.save(mask_file, image)

mask_path = '../../dataset' + '/bobo/mask'
mask2npy(1502, mask_path, 'mask_timg.jpeg')
# 目前有四种图案可以选择，也可以自己绘制图案，要显示的部分用白色，其余部分用黑色，图案在靠右偏上的位置，550*220*3，分辨率96dpi