import numpy as np
import cv2

import content_trans
import gen_step
import outline_get

SHAPE = (384, 384, 1)


def convert_arc(pt1, pt2, sagitta):
    # extract point coordinates
    x1, y1 = pt1
    x2, y2 = pt2

    # find normal from midpoint, follow by length sagitta
    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n ** 2))

    if np.isclose(n_dist, 0):
        # catch error here, d(pt1, pt2) ~ 0
        print('Error: The distance between pt1 and pt2 is too small.')

    n = n / n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2)) / 2 + sagitta * n

    # calculate the circle from three points
    # see https://math.stackexchange.com/a/1460096/246399
    A = np.array([
        [x1 ** 2 + y1 ** 2, x1, y1, 1],
        [x2 ** 2 + y2 ** 2, x2, y2, 1],
        [x3 ** 2 + y3 ** 2, x3, y3, 1]])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])

    if np.isclose(M11, 0):
        # catch error here, the points are collinear (sagitta ~ 0)
        print('Error: The third point is collinear.')

    cx = 0.5 * M12 / M11
    cy = -0.5 * M13 / M11
    radius = np.sqrt(cx ** 2 + cy ** 2 + M14 / M11)

    # calculate angles of pt1 and pt2 from center of circle
    pt1_angle = 180 * np.arctan2(y1 - cy, x1 - cx) / np.pi
    pt2_angle = 180 * np.arctan2(y2 - cy, x2 - cx) / np.pi

    return (cx, cy), radius, pt1_angle, pt2_angle


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=1, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
    # taken from https://stackoverflow.com/a/44892317/5087436
    center = (
        int(round(center[0] * 2 ** shift)),
        int(round(center[1] * 2 ** shift))
    )
    axes = (
        int(round(axes[0] * 2 ** shift)),
        int(round(axes[1] * 2 ** shift))
    )
    cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)
    return img


def clean():
    image = np.full(shape=SHAPE, fill_value=255, dtype=np.uint8)
    return image


def draw(image, p1, p2, sag, thick, colour):
    if sag == 0:
        line = np.full(shape=SHAPE, fill_value=255, dtype=np.uint8)  # line only
        line = cv2.line(line, p1, p2, color=colour, thickness=thick)
        image = cv2.line(image, p1, p2, color=colour, thickness=thick)
        return line, image
    else:
        arc = np.full(shape=SHAPE, fill_value=255, dtype=np.uint8)  # arc only
        center, radius, start_angle, end_angle = convert_arc(p1, p2, sag)
        axes = (radius, radius)
        image = draw_ellipse(image, center, axes, 0, start_angle, end_angle, color=colour, thickness=thick)
        arc = draw_ellipse(arc, center, axes, 0, start_angle, end_angle, color=colour, thickness=thick)
        return arc, image


def make_trans(content, style):
    img = content_trans.convert(content, style)
    if type(img) != np.ndarray:
        img = img.numpy()
    return img


def outline(original_image):
    output = outline_get.get_outline(original_image)
    return output


def firstline(original_image):
    img = clean()
    pred = gen_step.get_firstline(original_image)
    prev = pred
    drew = draw(img, pred[0:2], pred[2:4], pred[4], pred[5], pred[6])
    return pred, prev, drew


def nextline(original_image, prev, drew):
    img = clean()
    pred = gen_step.get_firstline(original_image)
    prev = pred
    drew = draw(img, pred[0:2], pred[2:4], pred[4], pred[5], pred[6])
    return pred, prev, drew
