from keras.preprocessing import image
import os

source_path = '../../dataset/bobo/raw/'
target_path = '../../dataset/bobo/cropped/'
resize_path = '../../dataset/bobo/resize/'

def crop_img():
    for i, image_name in enumerate(sorted(os.listdir(source_path))):
        image_path = os.path.join(source_path, image_name)
        x = image.load_img(image_path)
        x = image.img_to_array(x)

        camera_id = i//264 + 7
        location_id = (i%24)//2
        face_id = i%2 #表示脸的朝向 0朝前，1朝后
        person_id = (i%264)//24+1502

        if i < 264:
            if i % 24 == 0 or i % 24 == 1:
                x = x[590:1030, 660:800, :]
            elif i % 24 == 2 or i % 24 == 3:
                x = x[600:990, 570:680, :]
            elif i % 24 == 4 or i % 24 == 5:
                x = x[610:950, 490:600, :]
            elif i % 24 == 6 or i % 24 == 7:
                x = x[610:930, 430:550, :]
            elif i % 24 == 8 or i % 24 == 9:
                x = x[580:980, 990:1120, :]
            elif i % 24 == 10 or i % 24 == 11:
                x = x[600:940, 880:990, :]
            elif i % 24 == 12 or i % 24 == 13:
                x = x[610:920, 780:890, :]
            elif i % 24 == 14 or i % 24 == 15:
                x = x[620:900, 700:815, :]
            elif i % 24 == 16 or i % 24 == 17:
                x = x[600:920, 1230:1340, :]
            elif i % 24 == 18 or i % 24 == 19:
                x = x[610:900, 1110:1210, :]
            elif i % 24 == 20 or i % 24 == 21:
                x = x[610:880, 1010:1110, :]
            elif i % 24 == 22 or i % 24 == 23:
                x = x[620:860, 910:1010, :]
        elif 264 <= i < 528:
            if i % 24 == 0 or i % 24 == 1:
                x = x[540:1060, 400:570, :]
            elif i % 24 == 2 or i % 24 == 3:
                x = x[560:990, 470:640, :]
            elif i % 24 == 4 or i % 24 == 5:
                x = x[560:940, 540:690, :]
            elif i % 24 == 6 or i % 24 == 7:
                x = x[570:910, 580:720, :]
            elif i % 24 == 8 or i % 24 == 9:
                x = x[540:1060, 840:1060, :]
            elif i % 24 == 10 or i % 24 == 11:
                x = x[560:980, 860:1060, :]
            elif i % 24 == 12 or i % 24 == 13:
                x = x[560:940, 880:1040, :]
            elif i % 24 == 14 or i % 24 == 15:
                x = x[570:900, 890:1030, :]
            elif i % 24 == 16 or i % 24 == 17:
                x = x[530:1060, 1360:1550, :]
            elif i % 24 == 18 or i % 24 == 19:
                x = x[560:990, 1290:1480, :]
            elif i % 24 == 20 or i % 24 == 21:
                x = x[560:940, 1250:1430, :]
            elif i % 24 == 22 or i % 24 == 23:
                x = x[570:900, 1210:1350, :]
        elif i >= 528:
            if i % 24 == 0 or i % 24 == 1:
                x = x[570:920, 650:780, :]
            elif i % 24 == 2 or i % 24 == 3:
                x = x[570:900, 770:890, :]
            elif i % 24 == 4 or i % 24 == 5:
                x = x[570:860, 890:990, :]
            elif i % 24 == 6 or i % 24 == 7:
                x = x[580:840, 970:1080, :]
            elif i % 24 == 8 or i % 24 == 9:
                x = x[540:980, 920:1060, :]
            elif i % 24 == 10 or i % 24 == 11:
                x = x[550:920, 1030:1160, :]
            elif i % 24 == 12 or i % 24 == 13:
                x = x[560:900, 1130:1250, :]
            elif i % 24 == 14 or i % 24 == 15:
                x = x[570:870, 1210:1320, :]
            elif i % 24 == 16 or i % 24 == 17:
                x = x[520:1020, 1290:1470, :]
            elif i % 24 == 18 or i % 24 == 19:
                x = x[530:970, 1390:1550, :]
            elif i % 24 == 20 or i % 24 == 21:
                x = x[540:930, 1450:1630, :]
            elif i % 24 == 22 or i % 24 == 23:
                x = x[560:890, 1500:1630, :]

        x = image.array_to_img(x)
        image_name = '{:0>4}_c{}l{:0>2}f{}.JPG'.format(person_id,camera_id,location_id,face_id)
        image_path = os.path.join(target_path, image_name)
        image.save_img(image_path, x)

#交换image的名字；具体的，把str1开头的img，与str2开头的image交换
def rename(str1, str2):
    s_img = []
    s_img_name = []
    t_img = []
    t_img_name = []

    for image_name in sorted(os.listdir(target_path)):
        if image_name.startswith(str1):
            image_path = os.path.join(target_path, image_name)
            x = image.load_img(image_path)
            x = image.img_to_array(x)
            s_img.append(x)
            s_img_name.append(image_name)
        elif image_name.startswith(str2):
            image_path = os.path.join(target_path, image_name)
            x = image.load_img(image_path)
            x = image.img_to_array(x)
            t_img.append(x)
            t_img_name.append(image_name)

    for i in range(len(s_img)):
        image_path1 = os.path.join(target_path, t_img_name[i])
        x = image.array_to_img(s_img[i])
        image.save_img(image_path1, x)

        image_path2 = os.path.join(target_path, s_img_name[i])
        x = image.array_to_img(t_img[i])
        image.save_img(image_path2, x)

def max_h_w():
    h = 0
    w = 0
    for image_name in sorted(os.listdir(target_path)):
        image_path = os.path.join(target_path, image_name)
        x = image.load_img(image_path)
        w_tmp, h_tmp = x.size
        if h_tmp > h:
            h = h_tmp
        if w_tmp > w:
            w = w_tmp
    print(w,h) #220 530

#reszie all the cropped image into (550, 220)
def resize_img():
    for image_name in sorted(os.listdir(target_path)):
        image_path = os.path.join(target_path, image_name)
        x = image.load_img(image_path,target_size=(550, 220))
        x = image.img_to_array(x)
        x = image.array_to_img(x)
        image_path = os.path.join(resize_path, image_name)
        image.save_img(image_path, x)



#crop_img()
#rename('1511_c9','1512_c9')
resize_img()