from random import randrange

from flask.json import jsonify
from flask import Flask, render_template
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

UPLOAD_FOLDER = r''   # 上传路径
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','doc','zip','rar'])   # 允许上传的文件类型

app = Flask(__name__, static_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
res = None
process=0

# def line_base() -> Line:
#     line = (
#         Line()
#         .add_xaxis(["{}".format(i) for i in range(10)])
#         .add_yaxis(
#             series_name="",
#             y_axis=[0],
#             is_smooth=False,
#             label_opts=opts.LabelOpts(is_show=True),
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(title="动态数据"),
#             xaxis_opts=opts.AxisOpts(type_="value"),
#             yaxis_opts=opts.AxisOpts(type_="value"),
#         )
#     )
#     return line
def line_base() -> Line:
    line = (
        Line()
        .add_xaxis(["{}".format(i) for i in range(10)])
        .add_yaxis(
            series_name="",
            y_axis=[randrange(50, 80) for _ in range(10)],
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="动态数据"),
            xaxis_opts=opts.AxisOpts(type_="value"),
            yaxis_opts=opts.AxisOpts(type_="value"),
        )
    )
    return line

@app.route("/")
def index():
    return render_template("index2.html")


@app.route("/lineChart")
def get_line_chart():
    c = line_base()
    return c.dump_options_with_quotes()


idx = 1

@app.route("/lineDynamicData")
def update_line_data():
    global idx
    idx = idx + 1
    return jsonify({"name": idx, "value": randrange(50, 80)})
# @app.route("/lineDynamicData")
# def update_line_data():
#     global idx
#     global res
#     idx = idx + 1
#     # 在主线程中打印
#     # while True:
#     if res.not_empty:
#         tmp_res = res.get()
#         print(tmp_res)
#         time.sleep(1)
#     return jsonify({"name": idx, "value": (format(tmp_res[1],'.3f'))})

def allowed_file(filename):   # 验证上传的文件名是否符合要求，文件名必须带点并且符合允许上传的文件类型要求，两者都满足则返回 true
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':   # 如果是 POST 请求方式
        file = request.files['file']   # 获取上传的文件
        if file and allowed_file(file.filename):   # 如果文件存在并且符合要求则为 true
            filename = secure_filename(file.filename)   # 获取上传文件的文件名
            file_path=os.path.join("test_upload/", filename)
            file.save(file_path)   # 保存文件
            # zipfile解压
            z = zipfile.ZipFile(file_path, 'r')
            z.extractall("test_upload/")
            z.close()
            return '{} upload successed!'.format(filename)   # 返回保存成功的信息
    # 使用 GET 方式请求页面时或是上传文件失败时返回上传文件的表单页面
    return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form action="" method=post enctype=multipart/form-data>
              <p><input type=file name=file>
                 <input type=submit value=Upload>
            </form>
            '''

@app.route("/post_select_mask", methods=['GET', 'POST'])
def post_select_mask():
    global process
    process=1
    print(process)
    return "yes"


def adv_start():
    global res
    graph = tf.get_default_graph()
    # with tf.variable_scope('base_model'):
    net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
    probe_path = '../../dataset' + '/bobo/resize'
    mask_path = '../../dataset' + '/bobo/mask'
    adv_path = '../../dataset' + '/bobo/tar_adv'
    noise_path = '../../dataset' +'/bobo/tar_noise'


    adv_id = 1502
    adv_index = np.array([30, 24, 28, 32, 40, 44, 46,  68, 50,54, 58, 62, 70])
    target_id = 1504
    target_index = np.array([144,146,148,150,152,154])

    res = queue.Queue()

    # 创建消息队列, 对attack与adv函数进行改写添加queue参数


    # 创建攻击线程，然后在主线程中对结果进行获取
    attack_thread = threading.Thread(target=attack, args=(graph, res, net, probe_path, mask_path, adv_path, noise_path,
           adv_index, target_index, adv_id, target_id, 2,))
    attack_thread.start()



if __name__ == "__main__":
    # adv_start()
    app.run()