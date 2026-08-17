import torch

class WeightedHeatmapMSELoss(torch.nn.Module):
    def __init__(self, pos_weight=20.0, sigmoid=True):
        super().__init__()
        self.pos_weight = pos_weight
        self.sigmoid = sigmoid

    def forward(self, pred, target):
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        weight = 1.0 + self.pos_weight * target
        return (weight * (pred - target) ** 2).mean()