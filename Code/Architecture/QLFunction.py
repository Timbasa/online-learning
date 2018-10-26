import torch
import torch.nn as nn

class QuantileLossFunction(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles if isinstance(quantiles,list) else [quantiles]

    def forward(self, preds, target):
        # assert not target.requires_grad
        # assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds
            losses.append(torch.max((1 - q) * errors, -q*errors).unsqueeze(1))
        sum = torch.sum(torch.cat(losses, dim=1), dim=1)
        loss = sum/len(self.quantiles)
        return loss