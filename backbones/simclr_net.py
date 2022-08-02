import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

torch.manual_seed(0)


class SimCLRNet(nn.Module):

    def __init__(self, network, batch_size, T=0.07):
        super(SimCLRNet, self).__init__()
        self.encoder_q = network
        self.T = T
        self.batch_size = batch_size

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        k = self.encoder_q(im_k)
        features = torch.cat([q,k], dim=0)
        return self.info_nce_loss(features)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0) # 一般做两次增广，所以这里写死为2
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.T
        return logits, labels
    