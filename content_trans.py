import functools
import os

import keras.models

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import gridspec


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_base64, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    image_base64 = image_base64.replace("+", "-").replace("/", "_")
    img = tf.io.decode_image(
        tf.io.decode_base64(image_base64),
        channels=3,
        dtype=tf.float32
    )[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()


def convert(content, style):
    output_image_size = 384

    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content, content_img_size)
    style_image = load_image(style, style_img_size)

    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    # show_n([content_image, style_image], ['Content image', 'Style image'])

    # using normal tensorflow
    model = keras.models.load_model("models/style_transfer/normal")
    outputs = model(content_image, style_image)
    stylized_image = outputs[0]

    # show_n([content_image, style_image, stylized_image],
    #        titles=['Original content image', 'Style image', 'Stylized image'])
    return stylized_image[0]
