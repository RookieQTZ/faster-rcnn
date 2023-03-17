import torch


def load(loss_weight_file):
    with open(loss_weight_file, 'r') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行
        w1, w2, w3, w4 = lines[-1].strip().split(' ')  # 取最后一行
        w1, w2, w3, w4 = float(w1), float(w2), float(w3), float(w4)  # 将字符串类型转为可计算的float类型
        last_loss = torch.tensor([w1, w2, w3, w4])
    return last_loss


def save(loss_weight, loss_weight_file):
    # write into txt
    with open(loss_weight_file, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        weight_list = [weight.item() for weight in loss_weight]
        txt = " ".join('%s' % weight for weight in weight_list)
        f.write(txt + "\n")
