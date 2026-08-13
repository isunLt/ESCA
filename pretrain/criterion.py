import torch
from torch import nn


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss


class SupConLoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, l):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.eq(l.view(-1, 1), l).float()
        # target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        logits_max, _ = torch.max(out, dim=-1, keepdim=True)
        out = out - logits_max.detach()
        exp_out = torch.exp(out)
        num_p = torch.sum(target, dim=-1)
        denominator = torch.sum(exp_out, dim=-1, keepdim=True)
        log_probs = out - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * target, dim=-1) / num_p
        return -log_probs.mean()


class SupConLossIntra(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(SupConLossIntra, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, l):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.eq(l.view(-1, 1), l).float()
        # target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        logits_max, _ = torch.max(out, dim=-1, keepdim=True)
        out = out - logits_max.detach()
        exp_out = torch.exp(out)
        logits_mask = torch.ones_like(target, device=logits.device) - torch.eye(l.size(0), device=logits.device)
        p_m = target * logits_mask
        n_m = 1. - target
        num_p = torch.sum(p_m, dim=-1)
        denominator = torch.sum(exp_out * n_m, dim=-1, keepdim=True) + torch.sum(exp_out * p_m, dim=-1, keepdim=True)
        log_probs = out - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * target, dim=-1)[num_p > 0] / num_p[num_p > 0]
        return -log_probs.mean()


class SupConLossProto(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(SupConLossProto, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, l, l_p):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.eq(l.view(-1, 1), l_p).float()
        # target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        logits_max, _ = torch.max(out, dim=-1, keepdim=True)
        out = out - logits_max.detach()
        exp_out = torch.exp(out)
        num_p = torch.sum(target, dim=-1)
        denominator = torch.sum(exp_out, dim=-1, keepdim=True)
        log_probs = out - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * target, dim=-1) / num_p
        return -log_probs.mean()


class SupConLossTemporal(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(SupConLossTemporal, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, l, l_p):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.eq(l.view(-1, 1), l_p).float()
        # target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        # logits_max, _ = torch.max(out, dim=-1, keepdim=True)
        # out = out - logits_max.detach()
        exp_out = torch.exp(out)
        num_p = torch.sum(target, dim=-1)
        denominator = torch.sum(exp_out, dim=-1, keepdim=True)
        log_probs = out - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * target, dim=-1)[num_p > 0] / num_p[num_p > 0]
        return -log_probs.mean()


class SupConLossTemporalV2(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(SupConLossTemporalV2, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, l, l_p):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.eq(l.view(-1, 1), l_p).float()
        # target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        logits_max, _ = torch.max(out, dim=-1, keepdim=True)
        out = out - logits_max.detach()
        exp_out = torch.exp(out)
        logits_mask = torch.ones_like(target, device=logits.device)
        logits_mask[:l.size(0), :l.size(0)] -= torch.eye(l.size(0), device=logits.device)
        logits_mask[:, l.size(0):] = 0.
        p_m = target * logits_mask
        n_m = 1. - target
        num_p = torch.sum(p_m, dim=-1)
        denominator = torch.sum(exp_out * n_m, dim=-1, keepdim=True) + torch.sum(exp_out * p_m, dim=-1, keepdim=True)
        log_probs = out - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * p_m, dim=-1)[num_p > 0] / num_p[num_p > 0]
        return -log_probs.mean()
