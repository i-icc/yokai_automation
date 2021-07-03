from PIL import ImageGrab
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pyautogui
import sys

pyautogui.PAUSE = 0.03


def drag(points):
    if len(points) < 3:
        return
    pyautogui.mouseDown(points[0][0] // 2, points[0][1] // 2, button="left")
    points.pop(0)
    for point in points:
        pyautogui.moveTo(point[0] // 2, point[1] // 2, duration=0.07)
    pyautogui.mouseUp(points[-1][0] // 2, points[-1][1] // 2, button="left")


def keiro(data, flag, type, ind, pre, new):
    length = len(type)
    thisType = type[ind]
    flag[ind] = True
    thisPoint = (data[ind][0], data[ind][1])

    for i in range(length):
        if flag[i] or type[i] != thisType:
            continue
        x, y = data[i][0], data[i][1]
        point = (x, y)
        if distance.euclidean(point, thisPoint) < 170:
            new.append([x, y])
            pre = keiro(data, flag, type, i, pre, new)
            break
    if len(new) > len(pre):
        pre = new
    return pre


def printLine(img, data):
    for i in range(1, len(data)):
        cv2.line(
            img,
            (data[i - 1][0], data[i - 1][1]),
            (data[i][0], data[i][1]),
            (0, 0, 255),
        )


while True:
    img = ImageGrab.grab(
        bbox=(0, 0, 820, 1500)
    )  # bbox specifies specific region (bbox= x,y,width,height)
    frame = np.array(img)
    frame = cv2.medianBlur(frame, 5)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (33, 33), 1)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=50,
        param2=24,
        minRadius=30,
        maxRadius=70,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        clors = np.empty((len(circles[0, :]), 3))
        for i in range(len(circles[0, :])):
            crop = frame[
                circles[0][i][1]
                - (circles[0][i][2] * 3 // 5) : circles[0][i][1]
                + (circles[0][i][2] * 3 // 5),
                circles[0][i][0]
                - (circles[0][i][2] * 3 // 5) : circles[0][i][0]
                + (circles[0][i][2] * 3 // 5),
            ]
            try:
                b, g, r, a = cv2.split(crop)
            except:
                b, g, r, a = [0], [0], [0], [0]
            b_a = np.average(b)
            g_a = np.average(g)
            r_a = np.average(r)
            clors[i] = [b_a, g_a, r_a]
            cv2.circle(frame, (circles[0][i][0], circles[0][i][1]), 1, (0, 0, 255), 3)
        try:
            pred = KMeans(n_clusters=6).fit_predict(clors)
            road = []
            for i in range(len(circles[0, :])):

                cv2.putText(
                    frame,
                    str(pred[i]),
                    (circles[0][i][0], circles[0][i][1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 2, 255),
                    2,
                    cv2.LINE_AA,
                )
                flag = [False] * len(circles[0, :])
                road = keiro(
                    circles[0],
                    flag,
                    pred,
                    i,
                    road,
                    [[circles[0][i][0], circles[0][i][1]]],
                )
            printLine(frame, road)
            drag(road)
        except:
            pass
    cv2.imshow("preview", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
