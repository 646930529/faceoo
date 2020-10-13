import getface
import faceoo_gpu as faceoo
import cv2
import base64
import numpy as np
import sys
import gc
import zcode


def fs(x):
    return 1 / (1 + np.e ** (-x * 13 + 5)) * 100


def strtoimg(str):
    imgData = base64.b64decode(str)
    nparr = np.frombuffer(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def imgtostr(img):
    image = cv2.imencode('.jpg', img)[1]
    base64_data = base64.b64encode(image)
    return base64_data


import json
from flask import Flask, request, make_response

app = Flask(__name__)


@app.route("/getface", methods=["post"])
def getfacef():
    gc.collect()
    print('getface')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    img = strtoimg(j['base64'])
    print(img.shape)
    face, faces = getface.getface(img)
    if face is None:
        print('face is None')
        data = {
            'rtn': -1,
            'msg': 'no face'
        }
    else:
        print(face.shape)
        facestr = imgtostr(face)
        data = {
            'rtn': 0,
            'base64': facestr.decode()
        }
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


@app.route("/faceoo", methods=["post"])
def faceoof():
    gc.collect()
    print('faceoo')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    a = strtoimg(j['a'])
    b = strtoimg(j['b'])
    print(a.shape)
    print(b.shape)
    s = faceoo.faceoo(a, b)
    data = {
        'rtn': 0,
        's': float(s),
        'similarity': float(fs(s))
    }
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


@app.route("/faceoobig", methods=["post"])
def faceoobigf():
    gc.collect()
    print('faceoobig')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    a = strtoimg(j['a'])
    b = strtoimg(j['b'])
    print(a.shape)
    print(b.shape)

    a, a1 = getface.getface(a)
    if a is None:
        data = {'msg': 'a no face'}
    b, b1 = getface.getface(b)
    if b is None:
        data = {'msg': 'b no face'}
    print(a.shape)
    print(b.shape)

    if a is not None and b is not None:
        s = faceoo.faceoo(a, b)
        data = {
            'rtn': 0,
            's': float(s),
            'similarity': float(fs(s))
        }
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


@app.route("/getfacef", methods=["post"])
def getfaceff():
    gc.collect()
    print('getfacef')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    face = strtoimg(j['base64'])
    print(face.shape)
    f = faceoo.get_f(face)
    f = zcode.encode(f)
    data = {
        'rtn': 0,
        'f': f
    }
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


@app.route("/getbigf", methods=["post"])
def getbigff():
    gc.collect()
    print('getbigf')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    img = strtoimg(j['base64'])
    print(img.shape)
    face, faces = getface.getface(img)
    if face is None:
        print('face is None')
        data = {
            'rtn': -1,
            'msg': 'no face'
        }
    else:
        print(face.shape)
        f = faceoo.get_f(face)
        f = zcode.encode(f)
        data = {
            'rtn': 0,
            'f': f
        }
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


@app.route("/faceoof", methods=["post"])
def faceooff():
    gc.collect()
    print('faceoof')
    j = request.get_data()
    j = j.decode()
    j = json.loads(j)

    a = j['a']
    b = j['b']

    a = zcode.decode(a)
    b = zcode.decode(b)
    s = np.dot(a, b.T)
    data = {
        'rtn': 0,
        's': float(s),
        'similarity': float(fs(s))
    }
    print(data)
    rtnPath = make_response(json.dumps(data))
    rtnPath.headers['Access-Control-Allow-Origin'] = '*'
    return rtnPath


import time

if len(sys.argv) > 1:
    app.run(host="0.0.0.0", debug=False, port=int(sys.argv[1]))
else:
    app.run(host="0.0.0.0", debug=False, port=5000)
