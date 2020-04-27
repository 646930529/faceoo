import base64
import numpy as np


def encode(npa, dtype=np.float16):
    bs = npa.astype(dtype).tobytes()
    b64 = base64.b64encode(bs)
    b64 = 'zzqv1:' + b64.decode()
    return b64


def decode(txt, dtype=np.float16):
    b64 = txt[6:].encode()
    bs = base64.b64decode(b64)
    return np.frombuffer(bs, dtype).astype(np.float32)


# npa = np.random.rand(512)
# print(npa[:5])
# txt = encode(npa)
# print(len(txt))
# npa = decode(txt)
# print(npa[:5])
