from cv2 import cv2
import numpy as np
import time


def get_outline(image: np.ndarray, method="canny_blurred"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(gray, 50, 150)
    elif method == "canny_blurred":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_blurred = cv2.Canny(blurred, 30, 150)
        return canny_blurred


def group(image, sort_method="upper"):
    def dist(pa, pb):
        return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5

    def calculate_geocenter(dot_indexes):
        return tuple(s / len(dot_indexes) for s in list(map(sum, list(map(list, zip(*dot_indexes))))))

    def get_neighbor(coor, maps):
        checker = [(-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]
        oute = []
        for i in checker:
            target = (i[0] + coor[0], i[1] + coor[1])
            if target[0] < 0 or target[1] < 0 or target[0] >= maps.shape[0] or target[1] >= maps.shape[1]:
                continue
            if maps[target] != 0:
                oute.append(target)
                maps[target] = 0
        return oute, maps

    groups = []
    maps_all = image
    while (maps_all != 0).any():
        nextdo = [(np.where(maps_all != 0)[0][0], np.where(maps_all != 0)[1][0])]
        gp = {"geocenter": tuple, "dot_indexes": [nextdo[0]]}
        maps_all[nextdo[0]] = 0
        while nextdo:
            for n in list(nextdo.copy()):
                neighbors, maps_all = get_neighbor(n, maps_all)
                nextdo.remove(n)
                gp["dot_indexes"] += neighbors
                nextdo += neighbors
        gp["geocenter"] = calculate_geocenter(gp["dot_indexes"])
        groups.append(gp)
    if sort_method == "upper":
        return groups
    elif sort_method in ("nn", "nearest_neighbor"):
        sorted_group = [groups[0]]
        groups.pop(0)
        while groups:
            mini = min(groups, key=lambda d: dist(d["geocenter"], sorted_group[-1]["geocenter"]))
            sorted_group.append(mini)
            groups.remove(mini)
        return sorted_group
    elif sort_method in ("start_from_g", "sfg"):
        sorted_group = []
        while groups:
            mini = min(groups,
                       key=lambda d: dist(d["geocenter"], calculate_geocenter([f["geocenter"] for f in groups])))
            sorted_group.append(mini)
            groups.remove(mini)
        return sorted_group
    elif sort_method in ("far_from_g", "ffg"):
        sorted_group = []
        while groups:
            mini = max(groups,
                       key=lambda d: dist(d["geocenter"], calculate_geocenter([f["geocenter"] for f in groups])))
            sorted_group.append(mini)
            groups.remove(mini)
        return sorted_group
    elif sort_method in ("splitted_sort", "ss"):
        sorted_group = []



def show_image_sorted(o):
    for i in o:
        for index in i["dot_indexes"]:
            img[index] = 1
        cv2.imshow("", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    out = cv2.imread(r"stylized.jpg")
    out = cv2.resize(out, (384, 384))
    out = get_outline(out)
    img = np.zeros((384, 384))
    o = group(out, sort_method="nn")
    show_image_sorted(o)
    cv2.imwrite("outln.png", out)
