import os
import argparse
import tensorflow as tf
import sys
# sys.path.insert(1, '/models')
import models.model as model
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import time


def parse_args():

    parser = tools.argparser.parse_args([])
    parser.add_argument('--phase', type=str, default='test',
                        help='determine whether train or test')
    parser.add_argument('--datalist', type=str,
                        default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color',
                        help='model type: [lstm | gray | color]')
    parser.add_argument(
        '--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument(
        '--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4,
                        dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str,
                        default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='../pictures',
                        help='input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')
    args = parser.parse_args()
    return args

# @app.route('/')


def deblur1(path):
    args = {'model': 'color', 'datalist': 'C:/Users/smita/Desktop/face-group-recognition-facerecognition-api-updated/datalist_gopro.txt',
            'batch_size': 16, 'epoch': 4000, 'lr': 1e-4, 'gpu': '0', 'phase': 'test'}
    print(args['model'])
    # args = parse_args()

    # set gpu/cpu mode
    # if int(args.gpu_id) >= 0:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    if args['phase'] == 'test':
        out = "hello.png"
        res = deblur.test(720, 1280, path, out)
        path = 'static/Quality/'+str(time.time())+".png"
        cv2.imwrite(path, res)
        # print(x)
        return path
    elif args['phase'] == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


def contrast(path):
    img = cv2.imread(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L, kernel, iterations=1)
    max = cv2.dilate(L, kernel, iterations=1)
    min = min.astype(np.float64)
    max = max.astype(np.float64)
    contrast = (max-min)/(max+min)
    average_contrast = 100*np.nanmean(contrast)
    print(str(average_contrast)+"%")
    if float(average_contrast) < 3:
        return "low contrast"
    elif float(average_contrast) > 18:
        return "high contrast"
    else:
        return "contrast:natural"


def dark(path):
    img = cv2.imread(path)
    img_dot = img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    y, x, z = img.shape
    # print('>> Image Dimension => X:{}, Y:{}'.format(x, y))
    l_blur = cv2.GaussianBlur(l, (11, 11), 5)
    maxval = []
    count_percent = 3
    count_percent = count_percent/100
    row_percent = int(count_percent*x)
    column_percent = int(count_percent*y)
    for i in range(1, x-1):
        if i % row_percent == 0:
            for j in range(1, y-1):
                if j % column_percent == 0:
                    pix_cord = (i, j)

                    cv2.circle(img_dot, (int(i), int(j)), 5, (0, 255, 0), 2)
                    img_segment = l_blur[i:i+3, j:j+3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    # print('>> Total points: {}'.format(len(maxval)))
    # print('>> Average Brightness: {}'.format(avg_maxval))
    if avg_maxval < 20:
        return "dark"
    else:
        return "light"


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def blur(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < 100:
        return "blurry"
    else:
        return "not blurry"


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def convertContrastHigh(path):
    img = cv2.imread(path)
    # y = path.split('/')
    # imgName = y[8]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = adjust_gamma(img, 0.8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    outpath = 'static/Quality/'+str(time.time())+".png"
    cv2.imwrite(outpath, x)
    return outpath


def convertContrastLow(path):
    img = cv2.imread(path)
    # y = path.split('/')
    # imgName = y[8]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = adjust_gamma(img, 0.4)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    print("low contrast")
    outpath = 'static/Quality/'+str(time.time())+".png"
    cv2.imwrite(outpath,  x)
    return outpath


def convertdark(path):
    img = cv2.imread(path, 1)
    # y = path.split('/')
    # imgName = y[8]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    bl = clahe.apply(b)
    gl = clahe.apply(g)
    rl = clahe.apply(r)
    limg = cv2.merge((bl, gl, rl))
    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
    outpath = 'static/Quality/'+str(time.time())+".png"
    cv2.imwrite(outpath, limg)
    return outpath


if __name__ == '__main__':
    # app.run(port=4001,debug=True)
    tf.app.run()
