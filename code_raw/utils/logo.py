import numpy as np
import os
import scipy.misc
from matplotlib import pyplot as plt

logo_path = '../../dataset' + '/bobo'
noise_path = '../../dataset' +'/bobo/noise'
mask_path = '../../dataset' +'/bobo/mask'

logo_name = 'light.png'


pid=1502

mask = np.load(os.path.join(mask_path,'1502.npy')) #mask[140:290,90:185,:]=1

logo = scipy.misc.imread(os.path.join(logo_path, logo_name))
logo = scipy.misc.imresize(logo,(150,95))
x = (logo<30).astype(dtype=np.uint8)

mask = np.zeros((550,220,3),dtype=np.uint8)
mask[140:290,90:185,:] = x

np.save(os.path.join(mask_path,str(pid)+'logo.npy'), mask)

plt.subplot(121), plt.imshow(mask*255), plt.title('Input')
plt.show()

print(logo)