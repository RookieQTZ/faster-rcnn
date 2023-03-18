from torch import nn
import torch
from torch.nn import functional as F


# 二分类
# 实际没有用上，用的torchvision的focal loss
class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, reduction='mean'):
#         """
#         :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
#         :param gamma: 困难样本挖掘的gamma
#         :param reduction:
#         """
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             alpha = [0.25, 0.75]
#         self.alpha = torch.tensor(alpha)
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, pred, target):
#         target = target.long()
#         alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
#         pred = pred.view(pred.shape[0], 1)
#         log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
#         logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
#         logpt = logpt.view(-1)  # 降维，shape=(bs)
#         ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
#         pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
#         focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
#         if self.reduction == "mean":
#             return torch.mean(focal_loss)
#         if self.reduction == "sum":
#             return torch.sum(focal_loss)
#         return focal_loss


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#         super(FocalLoss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]
#         :return:
#         """
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1, preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
#         preds_logsoft = torch.log(preds_softmax)
#         labels = labels.long()
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#         loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
