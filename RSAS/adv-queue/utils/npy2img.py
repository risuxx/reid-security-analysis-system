# 该函数尝试将npy还原成图像
import numpy as np
import scipy.misc

a=np.load(r'E:\dataset\bobo\mask\1502.npy')
print(a.shape)
scipy.misc.imsave('outfile.jpg', a)
print(a)