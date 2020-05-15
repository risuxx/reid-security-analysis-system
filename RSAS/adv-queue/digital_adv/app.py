import shutil
from random import randrange

from flask.json import jsonify
from flask import Flask, render_template, make_response
from pyecharts import options as opts
from pyecharts.charts import Line

import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('../')
import queue
import threading
import time
from  target_bobo import attack
from flask import Flask, request
from werkzeug.utils import secure_filename   # 获取上传文件的文件名
import zipfile  #解压模块
from utils.img2npy import mask2npy #mask转化模块

UPLOAD_FOLDER = r''   # 上传路径
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','doc','zip','rar'])   # 允许上传的文件类型

app = Flask(__name__, static_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_path="../baseline_dis/market-pair-pretrain-withoutwarp.h5"
mask_path='../../dataset' + '/bobo/mask'
mask_name="Rectangle"

Loss_x =[] #x坐标轴
Loss_y =[] #y坐标轴
Dist_y=[]
adv_photo=['img/adv_affine/3.jpg', 'img/adv_affine/4.jpg',
           'img/adv_affine/1.jpg','img/adv_affine/2.jpg',
           'img/adv_affine/5.jpg','img/adv_affine/6.jpg']#展示的adv图片
adv_photo[0] = "/img/adv_affine/1502_c8l00f0.png"
adv_photo[1] = "/img/adv_affine/1502_c8l02f0.png"
adv_photo[2] = "/img/adv_affine/1502_c8l04f0.png"
adv_photo[3] = "/img/adv_affine/1502_c8l08f0.png"
adv_photo[4] = "/img/adv_affine/1502_c8l10f0.png"
adv_photo[5] = "/img/adv_affine/1502_c8l03f0.png"
res = None
process=0

# 根路径
@app.route("/")
def index():
    return render_template("index_initial.html")

# 验证上传的文件名是否符合要求的函数，文件名必须带点并且符合允许上传的文件类型要求，两者都满足则返回 true
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# #用于接收
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     # if request.method == 'POST':   # 如果是 POST 请求方式
#     #     file = request.files['file']   # 获取上传的文件
#     #     if file and allowed_file(file.filename):   # 如果文件存在并且符合要求则为 true
#     #         filename = secure_filename(file.filename)   # 获取上传文件的文件名
#     #         file_path=os.path.join("test_upload/", filename)
#     #         file.save(file_path)   # 保存文件
#     #         # zipfile解压
#     #         z = zipfile.ZipFile(file_path, 'r')
#     #         z.extractall("test_upload/")
#     #         z.close()
#     #         return '{} upload successed!'.format(filename)   # 返回保存成功的信息
#     # 使用 GET 方式请求页面时或是上传文件失败时返回上传文件的表单页面
#     return render_template("index2.html")
#     # return '''
#     #         <!doctype html>
#     #         <title>Upload new File</title>
#     #         <h1>Upload new File</h1>
#     #         <form action="" method=post enctype=multipart/form-data>
#     #           <p><input type=file name=file>
#     #              <input type=submit value=Upload>
#     #         </form>
#     #         '''
#上传model模型
@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    global model_path
    print("upload_model_开始")
    if request.method == 'POST':  # 如果是 POST 请求方式
        print("POST")
        file = request.files['file']  # 获取上传的文件
        if file and allowed_file(file.filename):  # 如果文件存在并且符合要求则为 true
            print("文件存在并且符合要求")
            filename = secure_filename(file.filename)  # 获取上传文件的文件名
            file_path = os.path.join("test_upload/", filename)
            file.save(file_path)  # 保存文件
            # zipfile解压
            print("zipfile解压,保存在upload_model/文件夹")
            z = zipfile.ZipFile(file_path, 'r')
            z.extractall("upload_model/")
            z.close()
            model_path="upload_model/model.h5"
            print(model_path)
            return jsonify({"success": 200})  # 返回失败的信息
            # return '{} upload successed!'.format(filename)  # 返回保存成功的信息
        return jsonify({"error": 1001, "msg": "上传失败"}) # 返回失败的信息
        # return jsonify({"success": 200})
    # return "失败"

#用于更改mask
@app.route("/post_select_mask", methods=['GET', 'POST'])
def post_select_mask():
    global mask_path
    global mask_name
    mask_name = request.values.get("mask") #获得mask
    print(mask_name)
    # mask_path = '../../dataset' + '/bobo/mask'
    if mask_name=="Android":
        mask2npy(1502, mask_path, 'mask_timg.jpeg')
        print("Android")
    elif mask_name=="QQ":
        mask2npy(1502, mask_path, 'mask_qq.jpg')
        print("QQ")
    elif mask_name == "Github":
        mask2npy(1502, mask_path, 'mask_tig.jpg')
        print("Github")
    else:  #默认为长方形
        mask2npy(1502, mask_path, 'outfile.jpg')
        print("Rectangle")
    # 目前有四种图案可以选择，也可以自己绘制图案，要显示的部分用白色，其余部分用黑色，图案在靠右偏上的位置，550*220*3，分辨率96dpi
    # process=1
    # print(process)
    return jsonify({"success": 200})  # 返回成功的信息

#启动模型
@app.route("/adv_start_submit")
def adv_start():
    global res
    global model_path
    global mask_path
    global mask_name
    graph = tf.get_default_graph()
    print("="*120)
    print("模型为："+model_path)
    print("蒙版为："+mask_name)
    # with tf.variable_scope('base_model'):
    # net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')
    # mask_path = '../../dataset' + '/bobo/mask'
    net = load_model(model_path)

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test' #
    probe_path = '../../dataset' + '/bobo/resize' #
    adv_path = '../../dataset' + '/bobo/tar_adv' #
    noise_path = '../../dataset' +'/bobo/tar_noise' #


    adv_id = 1502  #攻击者
    adv_index = np.array([30, 24, 28, 32, 40, 44, 46,  68, 50,54, 58, 62, 70])
    target_id = 1504 #目标
    target_index = np.array([144,146,148,150,152,154])

    # 创建消息队列, 对attack与adv函数进行改写添加queue参数
    res = queue.Queue()
    # 创建攻击线程，然后在主线程中对结果进行获取
    attack_thread = threading.Thread(target=attack, args=(graph, res, net, probe_path, mask_path, adv_path, noise_path,
           adv_index, target_index, adv_id, target_id, 2,))
    attack_thread.start()
    # while res.not_empty():
    #     time.sleep(1)
    #     print("res_empty")


    return jsonify({"success": 200})  # 返回成功的信息

########################################以上为预处理，以下为生成
#折线图
def line_base() -> Line:
    line = (
        Line()
        .add_xaxis(Loss_x)
        .add_yaxis(
            series_name="",
            y_axis=Loss_y,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="损失"),
            xaxis_opts=opts.AxisOpts(type_="value"),
            yaxis_opts=opts.AxisOpts(type_="value"),

        )
    )
    return line
def line_dist() -> Line:
    line = (
        Line()
        .add_xaxis(Loss_x)
        .add_yaxis(
            series_name="",
            y_axis=Dist_y,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="距离"),
            xaxis_opts=opts.AxisOpts(type_="value"),
            yaxis_opts=opts.AxisOpts(type_="value"),

        )
    )
    return line
#loss_chart
@app.route("/lineChart")
def get_line_chart():
    c = line_base()
    return c.dump_options_with_quotes()

#dist_char
@app.route("/dist_Chart")
def dist_Chart():
    c = line_dist()
    return c.dump_options_with_quotes()

idx = 0
# @app.route("/lineDynamicData")
# def update_line_data():
#     global idx
#     idx = idx + 1
#     return jsonify({"name": idx, "value": randrange(50, 80)})
# 这是调用bobo时要用的函数
@app.route("/lineDynamicData")
def update_line_data():
    global idx
    global res
    global Loss_x
    global Loss_y
    global Dist_y
    global adv_photo
    # 在主线程中打印
    # while True:

    if res.not_empty:
        tmp=res.get()
        if "adv" in tmp:
            print("==========================>adv in tmp")
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l00f0.png', 'templates/img/adv_affine/1502_c8l00f0.png')
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l02f0.png', 'templates/img/adv_affine/1502_c8l02f0.png')
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l04f0.png', 'templates/img/adv_affine/1502_c8l04f0.png')
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l08f0.png', 'templates/img/adv_affine/1502_c8l08f0.png')
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l10f0.png', 'templates/img/adv_affine/1502_c8l10f0.png')
            shutil.copyfile('../../dataset/bobo/tar_adv/1502_c8l03f0.png', 'templates/img/adv_affine/1502_c8l03f0.png')

            # adv_photo[0]="/img/adv_affine/1502_c8l00f0.png"
            # adv_photo[1]="/img/adv_affine/1502_c8l02f0.png"
            # adv_photo[2]="/img/adv_affine/1502_c8l04f0.png"
            # adv_photo[3]="/img/adv_affine/1502_c8l08f0.png"
            # adv_photo[4]="/img/adv_affine/1502_c8l10f0.png"
            # adv_photo[5]="/img/adv_affine/1502_c8l03f0.png"

            # return jsonify({"success": 200})
            # return jsonify({"error": 1001, "msg": "刷新"})  # 刷新图片
            # return render_template('index_generate.html', adv_photo1='img/adv_affine/3.jpg', adv_photo2='img/adv_affine/3.jpg',
            #                        adv_photo3='img/adv_affine/3.jpg',adv_photo4='img/adv_affine/3.jpg.jpg', adv_photo5='img/adv_affine/3.jpg',
            #                        adv_photo6='img/adv_affine/3.jpg')
            return jsonify({"end": 2})
            # return render_template('index_generate.html', adv_photo0=adv_photo[0], adv_photo1=adv_photo[1],
            #                        adv_photo2=adv_photo[2], adv_photo3=adv_photo[3], adv_photo4=adv_photo[4],adv_photo5=adv_photo[5])
        elif "advend" in tmp:
            print("==========================>adv end")
            return jsonify({"end": 1})
        else:
            idx = idx + 1
            print(tmp)
            Loss_x.append(idx) #保存在数组中
            Loss_y.append(format(tmp[5],'.3f')) #保存在数组中
            Dist_y.append(format(tmp[1]*10,'.1f'))
            # time.sleep(1)
            return jsonify({"end": 0, "name": idx, "value": (format(tmp[5],'.3f')),"value2": (format(tmp[1]*10,'.1f'))})


@app.route('/adv_run_photo_update', methods=['GET', 'POST'])
def adv_run_photo_update():
    import base64
    imageid = request.values.get("imageid") #获得imageid
    # print(imageid)
    img_local_path="templates/"+adv_photo[int(imageid)]
    # img_local_path="templates/img/adv_affine/1502_c8l03f0.png"
    f = open(img_local_path, 'rb')
    base64_str = base64.b64encode(f.read())
    # with open(img_local_path, 'r') as img_f:
    #     img_stream = img_f.read()
    #     img_stream = base64.b64encode(img_stream)
    # print(base64_str)
    print("==========================>adv_run_photo_update"+imageid)
    return jsonify({"status": 200,"base64_str":base64_str})

@app.route('/test_if',methods=['GET','POST'])
def test_if():
    imageid = request.values.get("imageid") #获得imageid
    print(imageid)
    if int(imageid) == 1:
        return jsonify({"end": 0})
    elif int(imageid) == 2:
        return jsonify({"end": 1})
    else:
        return jsonify({"end": 2})

@app.route("/adv_run")
def adv_run():
    return render_template('index_generate.html', adv_photo0=adv_photo[0], adv_photo1=adv_photo[1],
                           adv_photo2=adv_photo[2], adv_photo3=adv_photo[3],adv_photo4=adv_photo[4],adv_photo5=adv_photo[5])

########################################以上为生成，以下为评估
@app.route("/adv_evaluate")
def adv_evaluate():
    shutil.copyfile("noise.jpg", 'templates/img/adv_affine/noise.jpg')
    return render_template('index_evaluate.html', adv_photo0=adv_photo[0], adv_photo1=adv_photo[1],
                           adv_photo2=adv_photo[2], adv_photo3=adv_photo[3],adv_photo4=adv_photo[4],
                           adv_photo5=adv_photo[5],advPattern="/img/adv_affine/noise.jpg")

@app.route("/evaluate_calc")
def evaluate_calc():
    import random
    rank1=format(random.randint(0,100)/10,'.1f')
    rank5=format(random.randint(0,400)/10+float(rank1),'.1f')
    rank10=format(random.randint(0,500)/10+float(rank5),'.1f')
    mAP=(format(random.randint(100000,900000)/100000,'.6f'))
    return  jsonify({"rank1": rank1,"rank5": rank5,"rank10": rank10,"mAP": mAP})

@app.route("/evaluate_result")
def evaluate_result():
    import random
    # rank1=(format(random.randint(0,1),'.1f'))
    # rank5=(format(random.randint(0,5),'.1f'))
    # rank10=(format(random.randint(0,10),'.1f'))
    # mAP=(format(random.randint(100000,900000)/100000,'.6f'))
    evaluate_result="以上指标中，mAP低于理想值XX、rankn低于理想值XX，您的行人重识别模型存在很大的安全隐患"
    return  jsonify({"evaluate_result": evaluate_result})

@app.route("/download", methods=['GET'])
def download():
    # 此处的filepath是文件的路径，但是文件必须存储在static文件夹下， 比如images\test.jpg
    f = open("noise.jpg", 'rb')
    response = make_response(f.read())
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers['Content-Disposition'] = 'attachment;filename="{0}"'.format("noise.jpg")
    return response

if __name__ == "__main__":
    # adv_start()
    app.run()