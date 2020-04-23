import numpy as np
import scipy.misc
import os
import matplotlib.pyplot as plt


def load_raw(dir_path, img_names, index=None):
    if index is None:
        index = np.arange(len(img_names))

    image=[]
    infos = []

    for i in index:
        image_path = os.path.join(dir_path, img_names[i])
        x = np.array(scipy.misc.imread(image_path),dtype=np.float32)
        image.append(x)

        arr = img_names[i].split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    image = np.array(image)
    return image, infos


gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
img_names = sorted(os.listdir(gallery_path))

i = [13249]
imgs, infos = load_raw(gallery_path, img_names, i)
plt.imshow(imgs[0].astype(np.uint8))
plt.show()

scipy.misc.imsave(str(i[0])+'.png', imgs[0].astype(np.uint8))
