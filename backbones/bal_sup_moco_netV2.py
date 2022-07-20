# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class BalSupMoCoNet(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(BalSupMoCoNet, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("ba_queue", torch.randn(K, dim))
        self.register_buffer("nonba_queue", torch.randn(K, dim))
        self.ba_queue = nn.functional.normalize(self.ba_queue, dim=0)
        self.nonba_queue = nn.functional.normalize(self.nonba_queue, dim=0)

        self.register_buffer("ba_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("nonba_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):

        ba_ptr = int(self.ba_queue_ptr)
        nonba_ptr = int(self.nonba_queue_ptr)

        nonba_ids = torch.where(targets == torch.tensor([0]).cuda())[0]
        ba_ids = torch.where(targets == torch.tensor([1]).cuda())[0]

        # replace the keys at ptr (dequeue and enqueue)
        if self.K - ba_ptr >= len(ba_ids):
            self.ba_queue[ba_ptr:ba_ptr + len(ba_ids),:] = keys[ba_ids]
            ba_ptr = (ba_ptr + len(ba_ids)) % self.K  # move pointer
        else:
            bias = len(ba_ids) - (self.K - ba_ptr)
            self.ba_queue[ba_ptr:self.K,:] = keys[ba_ids[:self.K - ba_ptr]]
            self.ba_queue[:bias,:] = keys[ba_ids[self.K - ba_ptr:]]
            ba_ptr = bias

        if self.K - nonba_ptr >= len(nonba_ids):
            self.nonba_queue[nonba_ptr:nonba_ptr + len(nonba_ids),:] = keys[nonba_ids]
            nonba_ptr = (nonba_ptr + len(nonba_ids)) % self.K  # move pointer
        else:
            bias = len(nonba_ids) - (self.K - nonba_ptr)
            self.nonba_queue[nonba_ptr:self.K,:] = keys[nonba_ids[:self.K - nonba_ptr]]
            self.nonba_queue[:bias,:] = keys[nonba_ids[self.K - nonba_ptr:]]
            nonba_ptr = bias

        self.ba_queue_ptr[0] = ba_ptr
        self.nonba_queue_ptr[0] = nonba_ptr


    def forward(self, im_q, im_k, targets):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        qk_sim_matrix = torch.div(torch.matmul(q, k.T), self.T)
        mask_qk = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()

        qclass0_sim_matrix = torch.div(torch.matmul(q, self.ba_queue.clone().detach().T), self.T)
        # mask_qclass0 = (targets.unsqueeze(1) == torch.ones((1,self.K)).cuda()).float()
        mask_qclass0 = torch.zeros((targets.shape[0], self.K)).cuda()

        qclass1_sim_matrix = torch.div(torch.matmul(q, self.nonba_queue.clone().detach().T), self.T)
        # mask_qclass1 = (targets.unsqueeze(1) == torch.zeros((1,self.K)).cuda()).float()
        mask_qclass1 = torch.zeros((targets.shape[0], self.K)).cuda()

        sim_matrix = torch.cat([qk_sim_matrix, qclass0_sim_matrix, qclass1_sim_matrix], dim=1)
        mask = torch.cat([mask_qk, mask_qclass0, mask_qclass1], dim=1)

        log_prob = sim_matrix -  torch.log(torch.exp(sim_matrix).sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.mean()

        return loss

        # # compute logits
        # # Einstein sum is more intuitive
        # # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)

        # # apply temperature
        # logits /= self.T

        # # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()      

        # return logits, labels

