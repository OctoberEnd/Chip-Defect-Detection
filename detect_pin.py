import cv2
import numpy as np

src = cv2.imread("CQI-P/IC_33.png")

# 转换为灰度图像和二值图像
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, src_binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 形态学——开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dst = cv2.morphologyEx(src_binary, cv2.MORPH_OPEN, kernel)

# 距离变换
src_distance = cv2.distanceTransform(dst, cv2.DIST_L2, 3)
cv2.normalize(src_distance, src_distance, 0, 1, cv2.NORM_MINMAX)

# 距离变换——二值
_, src_distance = cv2.threshold(src_distance, 0.4, 1, cv2.THRESH_BINARY)
src_distance = np.uint8(src_distance * 255)

# 边缘检测和形态学处理
edges = cv2.Canny(src_distance, 80, 210)
result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel=(3, 5), iterations=10)

# 进行闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closeImage = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

# 寻找轮廓
contours, _ = cv2.findContours(closeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
marked_centers = 0  # 计数值
src_copy = np.copy(src, None)

# 在原始图像上绘制轮廓并标记轮廓中心点
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 55:
        continue

    marked_centers += 1  # 面积大于55的计数

    # 获取轮廓的最小外接矩形
    rect = cv2.minAreaRect(contour)
    center = tuple(map(int, rect[0]))

    # 在轮廓的中心点画绿色圆
    cv2.circle(src_copy, center, 5, (0, 255, 0), -1)

"""标记缺失"""

contours, hierarchy = cv2.findContours(closeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(closeImage)
point_list = []

for cnt in range(len(contours)):
    area = cv2.contourArea(contours[cnt])
    if area < 55:
        continue
    x, y, w, h = cv2.boundingRect(contours[cnt])
    cx = (x + w // 2)
    cy = (y + h // 2)

    point_list.append([cx, cy])
    cv2.circle(mask, (cx, cy), 2, (255), 2, 8, 0)

# 得到每个点的xy坐标后,分成两排
up_list = []
down_list = []
start_y = point_list[0][1]
for item in point_list:
    if abs(item[1] - start_y) <= 20:  # 同一排
        up_list.append(item)
    else:
        down_list.append(item)

# 对x坐标判断,是否有空缺
up_list = sorted(up_list, key=lambda x: x[0])  # 按照 x 坐标排序
down_list = sorted(down_list, key=lambda x: x[0])  # 按照 x 坐标排序
miss_pos1 = []  # 上排缺失的位置
miss_pos2 = []  # 下排缺失的位置


def judge(p_list, pos):
    miss_point = []
    gap = abs(p_list[0][0] - p_list[1][0])
    count = 0
    for i in range(len(p_list) - 1):
        this_gap = abs(p_list[i][0] - p_list[i + 1][0])
        if this_gap > (gap + 5):  # 间隔更大, 5是误差
            times = (this_gap + 5) // gap
            for j in range(times - 1):
                miss_x = p_list[i][0] + gap * (j + 1)
                miss_y = p_list[i][1]
                miss_point.append([miss_x, miss_y])
                if times == 2:
                    pos.append(i + 1 + j + 1 + count)  # 12 14
                else:
                    pos.append(i + 1 + j + 1)
                count += 1
    return miss_point


miss_list1 = judge(down_list, miss_pos1)
miss_list2 = judge(up_list, miss_pos1)


def draw_miss_point(src, miss_list):
    if miss_list:
        for item in miss_list:
            cv2.circle(src, (item[0], item[1]), 3, (0, 0, 255), 2, 8, 0)


draw_miss_point(src_copy, miss_list2)
draw_miss_point(src_copy, miss_list1)

if miss_list1:
    str = ','.join(map(str, miss_pos1))
    cv2.putText(src_copy, f"Top Row: No {str} pins missing", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(f"上排第{str}号引脚缺失")

if miss_list2:
    str = ','.join(map(str, miss_pos1))
    cv2.putText(src_copy, f"Bottom Row: No {str} pins missing", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(f"下排第{str}号引脚缺失")

cv2.imshow("draw red spot", src_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
