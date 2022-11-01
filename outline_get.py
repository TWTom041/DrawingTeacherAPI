import cv2
import numpy as np


def get_outline(image: np.ndarray, method="canny_blurred"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(gray, 30, 150)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_blurred = cv2.Canny(blurred, 30, 150)
    return canny_blurred


def group(image):
    def calculate_geocenter(dot_indexes):
        return tuple(s/len(dot_indexes) for s in list(map(sum, list(map(list, zip(*dot_indexes))))))

    def get_neighbor(coor, maps):
        checker = [(-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]
        oute = []
        for i in checker:
            target = (i[0] + coor[0], i[1] + coor[1])
            if target in maps:
                oute.append(target)
                maps.remove(target)
        return oute, maps

    def get_map(npimg):
        outee = []
        for i, p in np.ndenumerate(npimg):
            if p != 0:
                outee.append(i)
        return outee

    groups = []
    maps_all = get_map(image)
    while maps_all:
        nextdo = [maps_all[0]]
        gp = {"geocenter": tuple, "dot_indexes": [maps_all[0]]}
        maps_all.pop(0)
        while nextdo:
            for n in list(nextdo.copy()):
                neighbors, maps_all = get_neighbor(n, maps_all)
                nextdo.remove(n)
                gp["dot_indexes"] += neighbors
                nextdo += neighbors
        gp["geocenter"] = calculate_geocenter(gp["dot_indexes"])
        groups.append(gp)
    return groups


if __name__ == "__main__":
    out = cv2.imread(r"stylized.jpg")
    out = cv2.resize(out, (384, 384))
    out = get_outline(out, "canny")
    cv2.imwrite("canny.png", out)
    o = group(out)
    img = np.zeros((384, 384, 1))
    for a in o:
        for i in a["dot_indexes"]:
            img[i[0], i[1]] = 1
        cv2.imshow("", img)
        cv2.waitKey(0)
    cv2.imwrite("outln.png", out)
