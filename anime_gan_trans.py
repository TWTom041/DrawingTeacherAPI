import cv2
import numpy as np
import onnxruntime as ort

device_name = 'cpu'  # ort.get_device()

if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32:  # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img


def load_test_data(input_image: np.ndarray):
    img0 = input_image.astype(np.float32)
    img = process_image(img0)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]


def convert(img, scale, model_name):
    session = ort.InferenceSession(f'models/animeGANv2/{model_name}.onnx', providers=providers)
    x = session.get_inputs()[0].name
    fake_img = session.run(None, {x: img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, (scale[1], scale[0]))
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)


def process(input_image, model_name):
    mat, scale = load_test_data(input_image)
    res = convert(mat, scale, model_name)
    return res


if __name__ == "__main__":
    test_img = cv2.imread("content.jpg")
    o = process(test_img, "Hayao_v2")
    cv2.imshow("", o)
    cv2.waitKey(0)
