import os

import cv2 as cv
import numpy as np
import math
import shutil


def enhance(file_path, filename, i, output_path):
    img_path = os.path.join(file_path, filename)
    image = cv.imread(img_path)

    txt_name = filename.split(".")[0] + ".txt"

    # 锐化
    kernel = np.array([[-1, 0, -1],
                       [0, 5, 0],
                       [-1, 0, -1]], np.float32)

    dst = cv.filter2D(image, -1, kernel)
    cv.imwrite(os.path.join(output_path, str(i) + ".jpg"), dst)
    new_txt_name = str(i) + ".txt"
    shutil.copyfile(os.path.join(file_path, txt_name), os.path.join(output_path, new_txt_name))
    i += 1

    # 模糊
    dst = cv.blur(image, (5, 5))
    cv.imwrite(os.path.join(output_path, str(i) + ".jpg"), dst)
    new_txt_name = str(i) + ".txt"
    shutil.copyfile(os.path.join(file_path, txt_name), os.path.join(output_path, new_txt_name))
    i += 1

    # 雾化
    # img_f = image / 255.0
    # (row, col, chs) = image.shape
    #
    # A = 0.5  # 亮度
    # beta = 0.08  # 雾的浓度
    # size = math.sqrt(max(row, col))  # 雾化尺寸
    # center = (row // 2, col // 2)  # 雾化中心
    # for j in range(row):
    #     for i in range(col):
    #         d = -0.04 * math.sqrt((j - center[0]) ** 2 + (i - center[1]) ** 2) + size
    #         td = math.exp(-beta * d)
    #         img_f[j][i][:] = img_f[j][i][:] * td + A * (1 - td)
    #
    # cv.imwrite("../data/test/enhance/fog_100011.jpg", img_f)

    # 高斯噪声
    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    # 设置高斯分布的均值和方差
    mean = 0
    # 设置高斯分布的标准差
    sigma = 25
    # 根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean, sigma, (h, w, c))
    # 给图片添加高斯噪声
    noisy_img = image + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    # 保存图片
    cv.imwrite(os.path.join(output_path, str(i) + ".jpg"), noisy_img)
    new_txt_name = str(i) + ".txt"
    shutil.copyfile(os.path.join(file_path, txt_name), os.path.join(output_path, new_txt_name))


if __name__ == '__main__':
    # train val
    mode = "val"
    file_path = "../data/enhance/" + mode
    output_path = r"../data/enhance/" + mode + "/res"
    filenames = os.listdir(file_path)
    # 文件名从 110000 开始
    i = 110000
    for filename in filenames:
        # 跳过txt文件
        if not filename.endswith(".jpg"):
            continue
        if os.path.isdir(os.path.join(file_path, filename)):
            continue
        # 实现图像锐化、噪声、模糊
        print(filename)
        enhance(file_path, filename, i, output_path)
        i += 3
