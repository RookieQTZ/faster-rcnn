import PIL.Image
import numpy
import numpy as np
from PIL import Image
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
                  fill='white',
                  font=font)
        left += text_width


'''
根据目标边界框和紫外图像，评估放电严重状态，并在合成图中画出结果
'''
def evaluate(predict_boxes: numpy.ndarray, ul_org_path, ul_path, res_path):
    # 打开中值滤波处理过的紫外图片
    original_img = Image.open(ul_path)
    # 转成灰度图片
    img = original_img.convert('L')
    # 二值化
    threshold = 200
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    #  convert to binary image by the table
    binimg = img.point(table, "1")
    h = binimg.height
    w = binimg.width
    binimg = np.array(binimg, dtype=int)
    # 待评估图
    ul_org_img = PIL.Image.open(ul_org_path)
    for x1, y1, x2, y2 in predict_boxes:
        # 计算区域内光斑像素面积和区域面积
        area = (y2 - y1) * (x2 - x1)
        ul_area = 0
        for i in range(y1.astype(int), y2.astype(int)):
            for j in range(x1.astype(int), x2.astype(int)):
                if binimg[i, j] == 1:
                    ul_area += 1
        # 评估公式评估
        print("x1: " + str(x1) + ", ""x2: " + str(x2) + ", ""y1: " + str(y1) + ", ""y2: " + str(y2))
        print("area: " + str(area))
        print("spot area: " + str(ul_area))
        sle = ul_area / area
        res = ""
        color = "MediumAquamarine"
        if sle <= 0.05:
            res = "Normal"
            color = "MediumAquamarine"
        elif 0.05 < sle <= 0.15:
            res = "Primary"
            color = "LightSalmon"
        elif sle > 0.15:
            res = "Critical"
            color = "OrangeRed"
        # 在合成图上画出评估结果
        draw = ImageDraw.Draw(ul_org_img)
        # 绘制目标边界框
        draw.line([(x1, y1), (x1, y2), (x2, y2),
                   (x2, y1), (x1, y1)], width=3, fill=color)
        # 绘制类别和概率信息
        box = [x1, y1, x2, y2]
        draw_text(draw, box, res, color, 'arial.ttf', 20)
    plt.imshow(ul_org_img)
    plt.show()
    # 保存预测的图片结果
    ul_org_img.save(res_path)


if __name__ == '__main__':
    # 616 911 910 982
    # 724 979 815 1001
    evaluate(np.array([[724, 979, 815, 1001, ]], dtype=int),
             ul_org_path="../data/test/evaluate/pre/20m120kv.jpg",
             ul_path="../data/test/evaluate/pre/20m120kv_gray.jpg",
             res_path="../data/test/evaluate/pre/res/res.jpg")
