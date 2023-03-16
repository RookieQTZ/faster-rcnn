import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):

    def __init__(self, v_num=4):
        super(UncertaintyLoss, self).__init__()
        # sigma = torch.randn(v_num)
        # sigma = torch.empty((4, 1), dtype=torch.float32).uniform_(0.001, 0.95)
        sigma = torch.ones((4, 1), dtype=torch.float32)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        print("sigma: " + str(self.sigma))
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss

# if __name__ == '__main__':
#     weighted_loss_func = UncertaintyLoss(2)
#     weighted_loss_func.to(device)
#     optimizer = torch.optim.Adam(
#         filter(lambda x: x.requires_grad, list(model.parameters()) + list(weighted_loss_func.parameters())),
#         betas=(0.9, 0.98), eps=1e-09)
#
#     if epoch < 10:
#         loss = loss1
#     else:
#         loss = weighted_loss_func(loss1, loss2)
