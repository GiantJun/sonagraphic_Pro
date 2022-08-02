import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

torch.manual_seed(0)


class SupSimCLRNet(nn.Module):

    def __init__(self, network, batch_size, T=0.07):
        super(SupSimCLRNet, self).__init__()
        self.encoder_q = network
        self.T = T
        self.batch_size = batch_size

    def forward(self, im_q, im_k, targets):
        q = self.encoder_q(im_q)
        k = self.encoder_q(im_k)
        features = torch.cat([q,k], dim=0)
        return self.sup_contrastive_loss(features, targets)

    def sup_contrastive_loss(self, features, labels):

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != self.batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.div(torch.matmul(features, features.T), self.T)
        # for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        anchor_count, contrast_count = 2, 2 # 一般都是做两次增广，所以这里写死为2
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1, # 以行为单位替换处理
            torch.arange(self.batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )# 相当于单位矩阵取反，即对角线为0，其余为1
        mask = mask * logits_mask # 屏蔽掉自己与自己，将mask的对角线置为0

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # 屏蔽掉自己
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, self.batch_size).mean()

        return loss
    