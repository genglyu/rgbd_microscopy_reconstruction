import cv2
from copy import deepcopy


def img_add_text(string, img_bgr, pos=(100,100)):
    cv2.putText(img=img_bgr, text=string, org=pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)
    return img_bgr


def img_add_dot(img_bgr, pos, color=(0,0,255)):
    cv2.circle(img=img_bgr, center=pos, radius=2, color=color, thickness=1, lineType=cv2.LINE_AA, shift=0)
    return img_bgr


def img_add_cross_line(img_bgr, color=(0,255,0), thickness=1):
    (h, w, c) = img_bgr.shape
    cv2.line(img=img_bgr,
             pt1=(0, int(h / 2)), pt2=(w, int(h / 2)),
             color=color,
             thickness=thickness,
             lineType=cv2.LINE_AA,
             shift=0)
    cv2.line(img=img_bgr,
             pt1=(int(w / 2), 0), pt2=(int(w / 2), h),
             color=color,
             thickness=thickness,
             lineType=cv2.LINE_AA,
             shift=0)
    return img_bgr


