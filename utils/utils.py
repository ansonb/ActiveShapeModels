import json
import numpy as np
import cv2

class customJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, complex):
            return (obj.real, obj.imag)
        return json.JSONEncoder.default(self, obj)

def convertToGray(img):
    return cv2.cvt_color(img,cv2.COLOR_BGR2GRAY)

def decodeComplex(tensor):
    tensor = np.array(tensor)
    t1 = np.reshape(tensor,(-1,2))
    t2 = []
    for i, el in enumerate(t1):
        t2.append(complex(el[0],el[1]))
    t3 = np.reshape(t2,tensor.shape[:-1])
    return t3