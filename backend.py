import numpy as np
from cv2 import cv2
import base64

import content_trans
import outline_get
import anime_gan_trans


def b64tocv2(b64img):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64img), dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2tob64(npimg):
    return base64.b64encode(cv2.imencode('.png', npimg)[1]).decode()


def make_trans(content, style):
    if style not in ("Hayao", "Hayao_v2", "Paprika_v2", "Shinkai_v2"):
        img = content_trans.convert(b64tocv2(content), b64tocv2(style))
        if type(img) != np.ndarray:
            img = img.numpy()
        img *= 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(int)
    else:
        img = anime_gan_trans.process(b64tocv2(content), style)
        return img.astype(int)


def outline(original_image):
    output = outline_get.get_outline(original_image)
    return output


def gen_steps(outlines, sort_method, content_image=None):
    order = outline_get.group(outlines, sort_method=sort_method, content_image=content_image)
    return order

