import numpy as np
import cv2
import threading
import getface


insightface = cv2.dnn.readNetFromONNX('zface.onnx')
insightfaceLock = threading.Lock()


def normalize(arr, axis=1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(arr, order, axis))
    l2[l2 == 0] = 1
    return arr / np.expand_dims(l2, axis)


def get_feature(img):
    insightfaceLock.acquire()
    imgii = cv2.dnn.blobFromImage(img, size=(112, 112))
    insightface.setInput(imgii)
    embedding = insightface.forward()
    insightfaceLock.release()
    return normalize(embedding).flatten()


def get_f(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
