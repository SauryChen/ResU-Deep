def weighted_BCE(pred, target, weights = [0.5,1.5], reduction = 'sum'):
    if weights is not None:
        assert len(weights)==2
        loss = - weights[1] * (target * torch.log(pred)) - weights[0] * ((1-target) * torch.log(1-pred))
    else:
        loss = -(target * torch.log(pred)) - ((1-target) * torch.log(1-pred))
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
