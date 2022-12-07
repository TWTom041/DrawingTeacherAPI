from cv2 import cv2
import numpy as np
import onnxruntime as ort


def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.7:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > .25:
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes


def process(source):
    session = ort.InferenceSession("models/yolo/yolov5l-seg.onnx")
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source
    in_image = cv2.resize(resized, (640, 640))[:, :, ::-1].transpose(2, 0, 1) / 255.0
    in_image = np.expand_dims(in_image, axis=0).astype(np.float32)
    x = session.get_inputs()[0].name
    o = session.run(None, {x: in_image})[0][0]
    class_ids, confidences, boxes = unwrap_detection(resized, o)
    return class_ids, confidences, boxes


def get_foreground(image: np.ndarray, rect):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = rect.tolist()
    mask_new, b_model, f_model = cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    return mask_new


def kill_redundant(class_ids, confidences, boxes, threshold=10):
    new_class_ids = []
    new_confidences = []
    new_boxes = []
    for ci, co, bo in zip(class_ids, confidences, boxes):
        f = True
        for index, (nci, nco, nbo) in enumerate(zip(new_class_ids.copy(), new_confidences.copy(), new_boxes.copy())):
            if all(bo[i] - nbo[i] < threshold for i in range(4)):
                new_boxes[index] = np.array([
                    (bo[0] + nbo[0]) / 2,
                    (bo[1] + nbo[1]) / 2,
                    (bo[2] + nbo[2]) / 2,
                    (bo[3] + nbo[3]) / 2])
                f = False
        if f:
            new_class_ids.append(ci)
            new_confidences.append(co)
            new_boxes.append(bo)
    return new_class_ids, new_confidences, new_boxes


def get_mask(in_image):
    mask = np.zeros(in_image.shape[:2], dtype=np.uint8)
    class_ids, confidences, boxes = process(in_image)
    # boxes = [np.array([topx, topy, width, length])]
    class_ids, confidences, boxes = kill_redundant(class_ids, confidences, boxes, max(in_image.shape) / 20)
    # o = np.copy(in_image)
    for box in boxes:
        # (box[0], box[1]), (box[0]+box[2], box[1]+box[3])
        box = box.astype(int)
        mask = cv2.bitwise_or(mask, get_foreground(in_image, box))
        # o = cv2.rectangle(o, box, (255, 0, 0), 2)
    return mask


if __name__ == "__main__":
    a = cv2.imread("test.jpg")
    m = get_mask(a)

    cv2.imshow("m", m * 60)
    cv2.waitKey(0)
