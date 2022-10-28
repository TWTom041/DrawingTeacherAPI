import numpy as np
from scipy.interpolate import BSpline, splrep
from math import cos, atan, sin, pi
import matplotlib.pyplot as plt


def make_curve(x1, x2, y1, y2, theta, color):
    d = ((x1-x2)**2+(y1-y2)**2)**0.5
    x = np.array([x1, (x1+d/(2*cos(theta/4)))*cos(theta/4+atan((y1-y2)/(x1-x2))), x2])
    y = np.array([y1, (y1+d/(2*cos(theta/4)))*sin(theta/4+atan((y1-y2)/(x1-x2))), y2])
    tck = splrep(x, y, s=0, k=2)
    X_ = np.linspace(min(x), max(x), 500)
    Y_ = BSpline(*tck)(X_)
    plt.plot(X_, Y_)
    plt.show()


make_curve(0, 2, 0, 0, pi/2, "w")
