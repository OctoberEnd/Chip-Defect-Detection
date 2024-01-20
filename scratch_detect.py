import cv2
import numpy as np

minThres = 10

# 读取图像1
img1 = cv2.imread('CQI-S/NG/IC_22.png')
img2 = cv2.imread('CQI-S/NG/IC_22.png')

# 中值滤波
img1 = cv2.medianBlur(img1, 11)
# 图像差分
diff = cv2.absdiff(img1, img2)
# cv2.imshow('mid', img1)  # 结果图
# cv2.imshow('diff', diff)  # 结果图

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# 二值化
_, thres = cv2.threshold(gray, minThres, 255, cv2.THRESH_BINARY)
# cv2.imshow('thres', thres)

"""open"""
# 定义一个核（structuring element），这里使用矩形核
kernel = np.ones((1, 2), np.uint8)  # 可以根据需要调整核的大小

eroded_image = cv2.erode(thres, kernel, iterations=2)
dilated_image = cv2.dilate(eroded_image, kernel, iterations=3)

# cv2.imshow('opened_image', eroded_image)
# cv2.imshow('dilated_image', dilated_image)

# 查找轮廓
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i in range(0, len(contours)):
    length = cv2.arcLength(contours[i], True)
    # 通过轮廓长度筛选
    if length > 100:
        cv2.drawContours(img2, contours[i], -1, (0, 0, 255), 1)

cv2.imshow('result', img2)  # 结果图
cv2.waitKey(0)
cv2.destroyAllWindows()
