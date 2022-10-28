import cv2
import numpy as np

img = cv2.imread('stylized.jpg')


class line:
    points = []

    def append(self, point):
        self.points.append(point)


def get_outline(image: np.ndarray, method=None):
    method = method if method is not None else "canny_blurred"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(gray, 30, 150)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_blurred = cv2.Canny(blurred, 30, 150)
    return canny_blurred


def group(image: np.ndarray):
    points = np.argwhere(image != 0).tolist()
    line_group = []  # [{points: [], extreme: []}, ...]  extreme usually is
    while len(points) != 0:
        flag = True
        line_group.append({"points": [points[0]], "extreme": []})
        # get closest main loop
        while True:
            front_flag = False
            # left up is on left side
            # **.
            # *@.
            # *..
            for y in range(-1, 2):
                if [line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] - 1] in points:
                    front_flag = True
                    line_group[-1]["points"].insert(0, (line_group[-1]["points"][0][0] + y,
                                                        line_group[-1]["points"][0][1] - 1))
                    points.remove([line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] - 1])
            if [line_group[-1]["points"][0][0] - 1, line_group[-1]["points"][0][1]] in points:
                front_flag = True
                line_group[-1]["points"].insert(0, (line_group[-1]["points"][0][0] - 1,
                                                    line_group[-1]["points"][0][1]))
                points.remove([line_group[-1]["points"][0][0] - 1, line_group[-1]["points"][0][1]])
            # right bottom is on left side
            # ..*
            # .@*
            # .**
            for y in range(-1, 2):
                if [line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] + 1] in points:
                    front_flag = True
                    line_group[-1]["points"].insert(0, (line_group[-1]["points"][0][0] + y,
                                                        line_group[-1]["points"][0][1] + 1))
                    points.remove([line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] + 1])
            if [line_group[-1]["points"][0][0] + 1, line_group[-1]["points"][0][1]] in points:
                front_flag = True
                line_group[-1]["points"].insert(0, (line_group[-1]["points"][0][0] + 1,
                                                    line_group[-1]["points"][0][1]))
                points.remove([line_group[-1]["points"][0][0] + 1, line_group[-1]["points"][0][1]])
            end_flag = False
            # left up is on left side
            # **.
            # *@.
            # *..
            for y in range(-1, 2):
                if [line_group[-1]["points"][-1][0] + y, line_group[-1]["points"][-1][1] - 1] in points:
                    end_flag = True
                    line_group[-1]["points"].insert(len(line_group[-1]["points"]) - 1,
                                                    (line_group[-1]["points"][0][0] + y,
                                                     line_group[-1]["points"][0][1] - 1))
                    points.remove([line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] - 1])
            if [line_group[-1]["points"][0][0] - 1, line_group[-1]["points"][0][1]] in points:
                end_flag = True
                line_group[-1]["points"].insert(len(line_group[-1]["points"]) - 1,
                                                (line_group[-1]["points"][0][0] - 1,
                                                 line_group[-1]["points"][0][1]))
                points.remove([line_group[-1]["points"][0][0] - 1, line_group[-1]["points"][0][1]])
            # right bottom is on left side
            # ..*
            # .@*
            # .**
            for y in range(-1, 2):
                if [line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] + 1] in points:
                    end_flag = True
                    line_group[-1]["points"].insert(len(line_group[-1]["points"]),
                                                    (line_group[-1]["points"][0][0] + y,
                                                     line_group[-1]["points"][0][1] + 1))
                    points.remove([line_group[-1]["points"][0][0] + y, line_group[-1]["points"][0][1] + 1])
            if [line_group[-1]["points"][0][0] + 1, line_group[-1]["points"][0][1]] in points:
                end_flag = True
                line_group[-1]["points"].insert(len(line_group[-1]["points"]),
                                                (line_group[-1]["points"][0][0] + 1,
                                                 line_group[-1]["points"][0][1]))
                points.remove([line_group[-1]["points"][0][0] + 1, line_group[-1]["points"][0][1]])


if __name__ == "__main__":
    out = get_outline(cv2.imread(r"stylized.jpg"))
    group(out)
    cv2.imshow("", out)
    cv2.waitKey(0)
