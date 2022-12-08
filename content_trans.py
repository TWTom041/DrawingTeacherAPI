import onnxruntime
import numpy as np
from cv2 import cv2


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[0], shape[1])
    offset_y = max(shape[0] - shape[1], 0) // 2
    offset_x = max(shape[1] - shape[0], 0) // 2
    image = image[offset_y:(offset_y+new_shape), offset_x:(offset_x+new_shape)]
    return image.astype(np.float32)


# @functools.lru_cache(maxsize=None)
def load_image(npimage, image_size=(256, 256)):
    """Loads and preprocesses images."""
    img = npimage
    img = crop_center(img)
    img = (cv2.resize(img, image_size) / 255.)
    return img


def convert(content, style):
    content = content[:, :, ::-1]
    style = style[:, :, ::-1]
    output_image_size = 384

    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content, content_img_size)[np.newaxis, ...]
    style_image = load_image(style, style_img_size)
    style_image = cv2.blur(style_image, (3, 3))[np.newaxis, ...]
    session = onnxruntime.InferenceSession("models/style_transfer/fst.onnx")
    session.get_modelmeta()
    in_1 = session.get_inputs()[0].name
    in_2 = session.get_inputs()[1].name
    outs = session.get_outputs()[0].name
    outputs = session.run(None, {in_1: content_image, in_2: style_image})
    stylized_image = outputs[0]

    return stylized_image[0][:, :, ::-1]


if __name__ == "__main__":
    a = cv2.imread("content.jpg")
    b = cv2.imread("style.jpg")
    c = convert(a, b)
    cv2.imshow("", c)
    cv2.waitKey(0)
