import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from cv2 import cv2
import tensorflow as tf
import tensorflow_hub as hub


def get_semantic_segmentation(image: np.ndarray):
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512

    keras_layer = hub.KerasLayer('https://tfhub.dev/google/edgetpu/vision/deeplab-edgetpu/fused_argmax/s/1')
    model = tf.keras.Sequential([keras_layer])
    model.build([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    min_dim = min(image.shape[0], image.shape[1])
    image = cv2.resize(image,
                       (IMAGE_WIDTH * image.shape[0] // min_dim,
                        IMAGE_HEIGHT * image.shape[1] // min_dim))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(image, axis=0)
    input_data = input_data[:, :IMAGE_WIDTH, :IMAGE_HEIGHT, :]
    input_data = input_data.astype(np.float64) / 128 - 0.5

    output_data = model(input_data).numpy()[0]
    return output_data


if __name__ == "__main__":
    img = cv2.imread("t.jpg")
    o = get_semantic_segmentation(img)
    o = (o / o.max() * 255).astype(np.uint8)
    cv2.imshow("", o)
    cv2.waitKey(0)
