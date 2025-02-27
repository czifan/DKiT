import torch
import torch.nn as nn

class DKiTCriterion(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index) 

    def forward(self, pred, y):
        _, _, C = pred.shape
        pred = pred.view(-1, C)
        y = y.view(-1).long()
        loss = self.cross_entropy(pred, y)
        return loss 
    
class IdentityCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y):
        return torch.tensor(-1.0).to(pred.device)