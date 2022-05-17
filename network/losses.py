# This file contains code for different losses.
from typing import Any, Mapping, Text
import torch
import torch.nn.functional as F

def create_loss(loss_config: Mapping[Text, Any]) -> torch.nn.Module:
    """Creates a loss object based on the config.
    Args:
        loss_config (Mapping[Text, Any]): A mapping that specifies the loss.
    Returns:
        torch.nn.Module: The loss object.
    """
    loss_name = loss_config['TYPE']
    if loss_name == 'cross-entropy':
        return TopKCrossEntropy(loss_config['TOPK'], ignore_index=loss_config['IGNORE_INDEX'])
    elif loss_name == 'focal-loss':
        return TopKFocalLoss(loss_config['TOPK'], ignore_index=loss_config['IGNORE_INDEX'])
    else:
        raise ValueError(f'The specified loss {loss_name} is currently not supported.')


class TopKCrossEntropy(torch.nn.Module):
    """This class conmputes the top-k cross-entropy loss."""
    def __init__(self, k=1.0, ignore_index=-100):
        super(TopKCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.k = k

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=None,
                               ignore_index=self.ignore_index, reduction='none')
        if self.k == 1.0:
            return torch.mean(loss)
        else:
            loss = torch.flatten(loss, start_dim=1)
            valid_loss, _ = torch.topk(loss, int(self.k * loss.size()[1]))    
            return torch.mean(valid_loss)

class TopKFocalLoss(torch.nn.Module):
    """This class conmputes the top-k cross-entropy loss."""
    def __init__(self, k=1.0, alpha=0.25, gamma=2.0, ignore_index=-100):
        super(TopKFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.k = k
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=None,
                               ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * loss)
        if self.k == 1.0:
            return torch.mean(focal_loss)
        else:
            focal_loss = torch.flatten(focal_loss, start_dim=1)
            valid_loss, _ = torch.topk(focal_loss, int(self.k * focal_loss.size()[1]))    
            return torch.mean(valid_loss)