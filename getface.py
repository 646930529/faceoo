import cv2
import time
from itertools import product as product
from math import ceil

cfg = {
    'name': 'Retinaface',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

# parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.99, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.1, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# args = parser.parse_args()
args = {
    'cpu': False,
    'confidence_threshold': 0.99,
    'top_k': 5000,
    'nms_threshold': 0.1,
    'keep_top_k': 750,
    'vis_thres': 0.6
}


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


args = dict_to_object(args)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


import numpy
import onnxruntime as rt
import numpy as np


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    # print(priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:])
    # print(priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]))

    boxes = np.hstack((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])))
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


sess = rt.InferenceSession("FaceDetector.onnx")

resize = 1


def facesafe(rc):
    if rc is None:
        return None, None
    rc = rc[:4].astype(int)
    w = rc[2] - rc[0]
    h = rc[3] - rc[1]
    whc = 0
    b = int(max(w, h) * 0.2)

    if w > h:
        frc = [rc[0] - b, rc[1] - b - whc, rc[2] + b, rc[3] + b + whc]
    else:
        frc = [rc[0] - b - whc, rc[1] - b, rc[2] + b + whc, rc[3] + b]
    return frc

def detall(oimg):
    ow = oimg.shape[1]
    oh = oimg.shape[0]
    img_raw = cv2.resize(oimg, (640, 640), interpolation=cv2.INTER_AREA)
    im_height, im_width, _ = img_raw.shape
    scale = [img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]]
    img = img_raw.astype(np.float32)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.float32(img)[np.newaxis, :, :, :]

    pred_onx = sess.run(None, {'input0': img})

    tic = time.time()
    loc, conf = np.array(pred_onx[0]), np.array(pred_onx[1])
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors
    # print('net forward time: {:.4f}'.format(time.time() - tic))

    aaa = np.squeeze(loc.data)
    # print(aaa.shape)
    # print(prior_data.shape)
    # print(cfg['variance'])

    boxes = decode(aaa, prior_data, cfg['variance'])
    boxes = boxes * scale / resize

    scores = np.squeeze(conf)[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    # show image
    for b in dets:
        b[:4] = b[:4] * [ow / 640, oh / 640, ow / 640, oh / 640]
        b[0] = max(b[0], 0)
        b[1] = max(b[1], 0)
        b[2] = min(b[2], ow)
        b[3] = min(b[3], oh)
        b[:4] = facesafe(b[:4])
    return dets

def getborder(dets):
    border = 0
    if len(dets) > 0:
        m = np.array(dets).min()
        if m < 0:
            border = int(-m)
    return border

def det(oimg):
    ow = oimg.shape[1]
    oh = oimg.shape[0]
    img_raw = cv2.resize(oimg, (640, 640), interpolation=cv2.INTER_AREA)
    im_height, im_width, _ = img_raw.shape
    scale = [img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]]
    img = img_raw.astype(np.float32)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.float32(img)[np.newaxis, :, :, :]

    pred_onx = sess.run(None, {'input0': img})

    tic = time.time()
    loc, conf = np.array(pred_onx[0]), np.array(pred_onx[1])
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors
    # print('net forward time: {:.4f}'.format(time.time() - tic))

    aaa = np.squeeze(loc.data)
    # print(aaa.shape)
    # print(prior_data.shape)
    # print(cfg['variance'])

    boxes = decode(aaa, prior_data, cfg['variance'])
    boxes = boxes * scale / resize

    scores = np.squeeze(conf)[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    maxrc = None
    maxar = 0
    # show image
    for b in dets:
        b[:4] = b[:4] * [ow / 640, oh / 640, ow / 640, oh / 640]
        b[0] = max(b[0], 0)
        b[1] = max(b[1], 0)
        b[2] = min(b[2], ow)
        b[3] = min(b[3], oh)
        nowar = (b[2] - b[0]) * (b[3] - b[1])
        if nowar > maxar:
            maxar = nowar
            maxrc = b
    return maxrc


def draw(img, rc):
    if rc is not None:
        text = "{:.4f}".format(rc[4])
        b = list(map(int, rc))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


def getface(img):
    rc = det(img)
    if rc is None:
        return None, None
    s = rc[4]
    rc = rc[:4].astype(int)
    w = rc[2] - rc[0]
    h = rc[3] - rc[1]
    whc = int(abs(w - h) / 2)
    b = int(max(w, h) * 0.2)
    img = cv2.copyMakeBorder(img, b + whc, b + whc, b + whc, b + whc, cv2.BORDER_CONSTANT, value=0)
    rc = rc + b + whc

    if w > h:
        frc = [rc[0] - b, rc[1] - b - whc, rc[2] + b, rc[3] + b + whc]
    else:
        frc = [rc[0] - b - whc, rc[1] - b, rc[2] + b + whc, rc[3] + b]
    return img[frc[1]:frc[3], frc[0]:frc[2]], s


if __name__=='__main__':
    from glob import glob

    imgs = glob('./*.jpg')
    for img in imgs:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img, s = getface(img)
        if img is not None:
            cv2.imshow('asd', img)
            cv2.waitKey(0)



    from glob import glob

    imgs = glob('./*.jpg')
    for img in imgs:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        dets = detall(img)
        border = getborder(dets)
        img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        for d in dets:
            d = d + border
            cv2.rectangle(img, (d[0],d[1]), (d[2],d[3]), (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
