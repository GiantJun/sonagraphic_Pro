# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class BalSupMoCoNet(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, input_channel=3):
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

        if input_channel != 3:
            self.encoder_q.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder_k.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("ba_queue", torch.randn(K, dim))
        self.register_buffer("class0_queue", torch.randn(K, dim))
        self.class1_queue = nn.functional.normalize(self.class1_queue, dim=0)
        self.class0_queue = nn.functional.normalize(self.class0_queue, dim=0)

        self.register_buffer("class1_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("class0_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):

        class1_ptr = int(self.class1_queue_ptr)
        class0_ptr = int(self.class0_queue_ptr)

        class0_ids = torch.where(targets == torch.tensor([0]).cuda())[0]
        class1_ids = torch.where(targets == torch.tensor([1]).cuda())[0]

        # replace the keys at ptr (dequeue and enqueue)
        if self.K - class1_ptr >= len(class1_ids):
            self.class1_queue[class1_ptr:class1_ptr + len(class1_ids),:] = keys[class1_ids]
            class1_ptr = (class1_ptr + len(class1_ids)) % self.K  # move pointer
        else:
            bias = len(class1_ids) - (self.K - class1_ptr)
            self.class1_queue[class1_ptr:self.K,:] = keys[class1_ids[:self.K - class1_ptr]]
            self.class1_queue[:bias,:] = keys[class1_ids[self.K - class1_ptr:]]
            class1_ptr = bias

        if self.K - class0_ptr >= len(class0_ids):
            self.class0_queue[class0_ptr:class0_ptr + len(class0_ids),:] = keys[class0_ids]
            class0_ptr = (class0_ptr + len(class0_ids)) % self.K  # move pointer
        else:
            bias = len(class0_ids) - (self.K - class0_ptr)
            self.class0_queue[class0_ptr:self.K,:] = keys[class0_ids[:self.K - class0_ptr]]
            self.class0_queue[:bias,:] = keys[class0_ids[self.K - class0_ptr:]]
            class0_ptr = bias

        self.class1_queue_ptr[0] = class1_ptr
        self.class0_queue_ptr[0] = class0_ptr


    def forward(self, im_q, im_k, targets):
        """
        Input:
            im_q: a class1tch of query images
            im_k: a class1tch of key images
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
        
        # for k
        qk_sim_matrix = torch.div(torch.matmul(q, k.T), self.T)
        mask_qk = (targets.unsqueeze(0) == targets.unsqueeze(1)) # boolean
        diagonal_mask  = torch.scatter(
            torch.ones_like(mask_qk),
            1, # 以行为单位替换处理
            torch.arange(mask_qk.shape[0]).view(-1, 1).cuda(),
            0
        )# 相当于单位矩阵取反，即对角线为0，其余为1
        mask_qk = mask_qk * diagonal_mask # 去掉自身

        # for class0
        q_class0_sim_matrix = torch.div(torch.matmul(q, self.class0_queue.clone().detach().T), self.T)
        mask_q_class0 = (targets.unsqueeze(1) == torch.zeros((1,self.K)).cuda()) # boolean
        # for class1
        q_class1_sim_matrix = torch.div(torch.matmul(q, self.class1_queue.clone().detach().T), self.T)
        mask_q_class1 = (targets.unsqueeze(1) == torch.ones((1,self.K)).cuda()) # boolean

        
        # # for numerical stability
        # qk_sim_max = torch.max(qk_sim_matrix, dim=1, keepdim=True)[0]
        # qk_sim_matrix = qk_sim_matrix - qk_sim_max.detach()

        # q_class0_sim_max = torch.max(q_class0_sim_matrix, dim=1, keepdim=True)[0]
        # q_class0_sim_matrix = q_class0_sim_matrix - q_class0_sim_max.detach()

        # q_class1_sim_max = torch.max(q_class1_sim_matrix, dim=1, keepdim=True)[0]
        # q_class1_sim_matrix = q_class1_sim_matrix - q_class1_sim_max.detach()


        sim_matrix = torch.cat([qk_sim_matrix, q_class0_sim_matrix, q_class1_sim_matrix], dim=1)

        bank_mask = torch.cat([mask_q_class0, mask_q_class1], dim=1) # boolean

        # without_pos_bank_mask = torch.cat([torch.ones_like(mask_qk), (~bank_mask).float()], dim=1) # float, use negative in qk
        without_pos_bank_mask = torch.cat([mask_qk, (~bank_mask).float()], dim=1) # float, do not use negative in qk

        log_prob = qk_sim_matrix - torch.log((torch.exp(sim_matrix) * without_pos_bank_mask).sum(1))

        mean_log_prob_pos = (log_prob * mask_qk.float()).sum(1) / mask_qk.sum(1)

        loss = - mean_log_prob_pos.mean()

        self._dequeue_and_enqueue(k, targets)

        return loss

