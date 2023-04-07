import cv2
import os


# 读取该文件夹下所有的文件
path = "F:\desktop\\tmp"
output_path = "F:\desktop\\tmp\\res"
filelist = os.listdir(path)
# 遍历所有文件
for files in filelist:
    Olddir = os.path.join(path, files)  # 原来的文件路径
    if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        continue
    # os.path.splitext(path)  #分割路径，返回路径名和文件扩展名的元组
    # 文件名，此处没用到
    filename = os.path.splitext(files)[0]
    print(filename)
    img = cv2.imread(Olddir)
    h = img.shape[0]
    w = img.shape[1]
    h_resize = h // 4
    w_resize = w // 4
    img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_path, filename + ".jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
