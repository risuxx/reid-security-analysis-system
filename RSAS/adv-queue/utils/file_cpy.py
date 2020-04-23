import shutil
import os

def filecopy(src, dst):
    shutil.rmtree(dst)
    shutil.copytree(src, dst)

def rm_jpg(dir_path):
    for img_name in os.listdir(dir_path):
        if img_name.endswith('.png'):
            if os.path.exists(os.path.join(dir_path,img_name.replace('png','JPG'))):
                os.remove(os.path.join(dir_path,img_name.replace('png','JPG')))

if __name__ == '__main__':
    probe_path = '../../dataset' + '/bobo/resize'
    adv_path = '../../dataset' + '/bobo/untar_adv'

    wear_probe_path = '../../dataset' + '/wear2/resize'
    wear_adv_path = '../../dataset' + '/wear2/tar_adv'

    filecopy(wear_probe_path, wear_adv_path)
    #filecopy(probe_path, adv_path)
    #rm_jpg(adv_path)