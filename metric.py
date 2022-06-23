import torch

from typing import List
from sklearn.metrics import f1_score, accuracy_score


def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def relation_acc(output: torch.tensor, target: torch.tensor) -> List[torch.tensor]:
    '''
    Computes the accuracy for relation type predictions
    '''
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        correct = correct.float().sum()
        return correct.mul_(100.0 / batch_size)


def concept_metrics(output: torch.tensor, target: torch.tensor):
    '''
    Computes the accuracy and f1 score for concept type predictions
    '''
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.where(output > 0, 1.0, 0.0)
        correct = pred.eq(target)
        correct = correct.float().sum()
        acc = correct.mul_(100.0 / (batch_size * target.size(1)))
        f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='samples')
        return acc, f1
