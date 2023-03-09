import datetime
import matplotlib.pyplot as plt
import torch
from visdom import Visdom
import numpy as np


def create_visdom(visdom_file):
    viz = Visdom(env="demo", log_to_filename=visdom_file)
    return viz


def load_visdom(viz, visdom_file):
    Visdom.replay_log(viz, log_filename=visdom_file)


def visdom_draw(viz, value, epoch, title='title', ylabel='ylabel'):
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    else:
        value = [value]

    opts = {
        "title": title,
        "xlabel": 'epoch',
        "ylabel": ylabel,
        "width": 300,
        "height": 200,
        "legend": [ylabel]
    }

    viz.line(X=[epoch], Y=[value], win=title, update='append', opts=opts)


def visdom_pr(viz, coco_eval):
    pr_arr1 = coco_eval.eval['precision'][0, :, 0, 0, 2].tolist()
    pr_arr2 = coco_eval.eval['precision'][4, :, 0, 0, 2].tolist()
    pr_arr3 = coco_eval.eval['precision'][8, :, 0, 0, 2].tolist()

    opts = {
        "title": "Precision Recall Curve",
        "xlabel": 'recall',
        "ylabel": 'precision',
        "width": 300,
        "height": 200,
    }

    x = np.arange(0.0, 1.01, 0.01).tolist()

    viz.line(X=x, Y=pr_arr1, name="IOU=0.5", win="Precision Recall Curve", opts=opts)
    viz.line(X=x, Y=pr_arr2, name="IOU=0.7", win="Precision Recall Curve", opts=opts, update='append')
    viz.line(X=x, Y=pr_arr3, name="IOU=0.9", win="Precision Recall Curve", opts=opts, update='append')


def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)
