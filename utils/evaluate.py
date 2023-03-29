import PIL.Image
import numpy as np
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt


def draw_text(draw,
              box: list,
              res: str,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = res
    display_str_heights = font.getsize(display_str)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * display_str_heights

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


'''
根据目标边界框和紫外图像，评估放电严重状态，并在合成图中画出结果
'''
def evaluate(x1: int, y1: int, x2: int, y2: int, ul_org_path, ul_path, res_path):
    # 打开中值滤波处理过的紫外图片
    ul_img = cv2.imread(ul_path)
    # 转成灰度图片
    img = cv2.cvtColor(ul_img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binimg = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # 计算区域内光斑像素面积和区域面积
    area = (y2 - y1) * (x2 - x1)
    ul_area = 0
    h = binimg.shape[0]
    w = binimg.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            if binimg[i, j] == 255:
                ul_area += 1
    # 评估公式评估
    sle = ul_area / area
    res = ""
    if sle <= 0.05:
        res = "Normal"
    elif 0.05 < sle <= 0.15:
        res = "Primary"
    elif sle > 0.15:
        res = "Critical"
    # 在合成图上画出评估结果
    ul_org_img = PIL.Image.open(ul_org_path)
    draw = ImageDraw.Draw(ul_org_img)
    # 绘制目标边界框
    draw.line([(x1, y1), (x1, y2), (x2, y2),
               (x2, y1), (x1, y1)], width=3, fill="Yellow")
    # 绘制类别和概率信息
    box = [x1, y1, x2, y2]
    draw_text(draw, box, res, "Yellow", 'arial.ttf', 20)

    plt.imshow(ul_org_img)
    plt.show()
    # 保存预测的图片结果
    ul_org_img.save(res_path)


if __name__ == '__main__':
    evaluate(202, 277, 1031, 654,
             ul_org_path="../data/test/evaluate/ul+img.jpg",
             ul_path="../data/test/evaluate/med.jpg",
             res_path="../data/test/evaluate/res_normal.jpg")
