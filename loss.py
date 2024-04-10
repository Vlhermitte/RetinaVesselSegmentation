import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), f"y_pred: {y_pred.size()}, y_true: {y_true.size()}"
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)