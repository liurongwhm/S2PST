import torch
import torch.nn as nn
import torch.nn.functional as F
class self_training_conditional(nn.Module):
    def __init__(self, threshold: float):
        super(self_training_conditional, self).__init__()
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=0)

    def forward(self, td_pre, sd_pre):
        confidence_t, pseudo_labels_t = td_pre.max(dim=1)
        confidence_s, pseudo_labels_s = sd_pre.max(dim=1)

        mask_t = confidence_t > self.threshold
        mask_s = confidence_s > self.threshold

        if not (any(mask_t) and any(mask_s)):
            return torch.tensor(0.0).cuda(), mask_t, torch.tensor(0.0).cuda(), None

        mask_td_pre = td_pre[mask_t]
        mask_sd_pre = sd_pre[mask_s]
        filtered_s_label = pseudo_labels_s[mask_s]
        filtered_t_label = pseudo_labels_t[mask_t]
        sim = torch.matmul(mask_td_pre, mask_sd_pre.t())
        score = torch.tensor([
            sim[i, (filtered_s_label == filtered_t_label[i])].sum()
            for i in range(filtered_t_label.shape[0])
        ]).cuda()

        weights = self.softmax(score)
        loss = F.nll_loss(torch.log(mask_td_pre), filtered_t_label, reduction='none')
        weighted_loss = (loss * weights).mean()

        return weighted_loss, mask_t, pseudo_labels_t, weights
