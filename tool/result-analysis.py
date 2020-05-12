# !/usr/bin/env python
# -*- coding:utf-8 -*- 


import numpy as np

# 输入单应矩阵H和点pt，输出转换后的点
def HTransformPt(H, pt):
    pt_h = np.array([pt[0],pt[1],1]).reshape(3,1)
    dst_pt = H.dot(pt_h)
    dst_pt = dst_pt/dst_pt[2]
    return dst_pt[0:2]

# 计算两个单应矩阵的误差(8点法表示下的无穷范数)
def CalHomographyDiff(H1, H2, img_w, img_h):
    # 1、确定边界点
    H_w_c = np.linalg.inv(H1)
    h1_cross_h2 = np.cross(H_w_c[:,0], H_w_c[:,1])
    h1_cross_h2 = -h1_cross_h2 / h1_cross_h2[2]
    # print(h1_cross_h2)

    x = [0, img_w, img_w, 0]
    y = [0, 0, img_h, img_h]
    h_cross_0 = h1_cross_h2[0]
    h_cross_1 = h1_cross_h2[1]
    h_cross_2 = h1_cross_h2[2]
    # print('h_cross_0 = ', h_cross_0)
    lamda = []
    lamda_threshold = -0.7
    for i in range(0,4):
        lamda.append(h_cross_0 * x[i] + h_cross_1 * y[i] + h_cross_2)

    intersections = []
    intersections.append([(lamda_threshold - h_cross_2) / h_cross_0, 0])
    intersections.append([img_w, (lamda_threshold - h_cross_2 -h_cross_0 * img_w) /h_cross_1])
    intersections.append([(lamda_threshold - h_cross_2 - h_cross_1* img_h) /h_cross_0, img_h])
    intersections.append([0, (lamda_threshold - h_cross_2) /h_cross_1])

    # print(lamda)
    corners = []
    if lamda[0] < lamda_threshold:
        corners.append([x[0], y[0]])
    if intersections[0][0]< img_w and intersections[0][0] >= 0:
		corners.append(intersections[0])	
    if lamda[1] < lamda_threshold:
        corners.append([x[1], y[1]])
    if (intersections[1][1] < img_h and intersections[1][1] >= 0):
        corners.append(intersections[1])
    if lamda[2] < lamda_threshold:
        corners.append([x[2], y[2]])
    if intersections[2][0] < img_w and intersections[2][0] >= 0:
        corners.append(intersections[2])
    if lamda[3] < lamda_threshold:
        corners.append([x[3], y[3]])
    if intersections[3][1]< img_h and intersections[3][1] >= 0:
        corners.append(intersections[3])

    max_diff = 0
    for i in range(0,len(corners)):
        dst_pt_1 = HTransformPt(H1, np.array(corners[i]))
        dst_pt_2 = HTransformPt(H2, np.array(corners[i]))
        pt_diff = np.max(np.abs(dst_pt_1 - dst_pt_2))
        if max_diff < pt_diff:
            max_diff = pt_diff
    return max_diff


# 计算结果的精确度和召回率
# result - img_num | localization_state | time | h
# gr - 真值
def CalPrecisionAndRecall(result, gr, h_diff_threshold = 10):
    # cal homography difference
    N = result.shape[0]
    H_error = np.zeros([N,1])
    img_w = 1500.0
    img_h = 1500.0
    offset = np.array([[1,0,0],[0,1,213],[0,0,1]])
    for i in range(0,N):
        if result[i,1]==1:
            H_e = result[i,3:12].reshape(3,3)
            H_gr = offset.dot(gr[i,:].reshape(3,3))
            H_error[i] = CalHomographyDiff(H_e, H_gr, img_w, img_h)
        else:
            H_error[i] = result[i,1]
    # debug
    # print(H_error)
    print("Error localiazation:")
    for i in range(0, len(H_error)):
        if H_error[i]>h_diff_threshold:
            print(i+1, H_error[i,0])

    # calculate precision and recall
    right_num = len(H_error[np.bitwise_and(H_error > 0, H_error <= h_diff_threshold)])
    if len(H_error[H_error > 0]) == 0:
        precision = 0
    else:
        precision = float(right_num) / len(H_error[H_error > 0])
    
    if len(H_error[H_error != -1]) == 0:
        recall = 0
    else:
        recall = float(right_num) / len(H_error[H_error!=-1])

    return precision, recall

    

# data analysis
root_path = '/media/li/flight_sim/'
i = 0
area_num = 17
result_file = root_path + 'paper-1/result-{0}/{1}-result'.format(area_num, i)
# grid_size = 50
# model_h_error = 200
# result_file = root_path + 'paper-1/result-17/gs-{0}-e-{1}-result'.format(grid_size, model_h_error)
gr_file = root_path + 'image_data/view_angle_{0}/merge/{1}/h.txt'.format(area_num, i)

result = np.loadtxt(result_file)
gr = np.loadtxt(gr_file)

precision, recall = CalPrecisionAndRecall(result, gr, 500)
print('precision = {0}, recall = {1}'.format(precision, recall))
