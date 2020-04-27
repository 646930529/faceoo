import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import numpy as np
import cv2
import threading
import getface


def get_model():
    ctx = mx.gpu(0)
    sym, arg_params, aux_params = onnx_mxnet.import_model('zface.onnx')
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    return model


face_model = get_model()
insightfaceLock = threading.Lock()


def get_feature(aligned):
    insightfaceLock.acquire()
    data = mx.nd.array(np.expand_dims(aligned, axis=0))
    db = mx.io.DataBatch(data=(data,))
    face_model.forward(db, is_train=False)
    embedding = face_model.get_outputs()[0].asnumpy()
    insightfaceLock.release()
    return normalize(embedding).flatten()


def normalize(arr, axis=1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(arr, order, axis))
    l2[l2 == 0] = 1
    return arr / np.expand_dims(l2, axis)


def get_f(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    return get_feature(img)


def faceoo(a, b):
    a = get_f(a)
    b = get_f(b)

    return np.dot(a, b.T)


# a = cv2.imread('aa.jpg')
# b = cv2.imread('bb.jpg')
#
# a, a1 = getface.getface(a)
# b, b1 = getface.getface(b)
#
# print(faceoo(a, b))
