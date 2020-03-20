import torch
from torch import nn
import math
import torch.distributed as dist
from moco.util import dist_collect


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, batch_size, feature_dim, queue_size, temperature=0.07, num_classes=1000, k_shot=0, info_temperature=0.07):
        super(MemoryMoCo, self).__init__()
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.temperature = temperature
        self.num_classes = num_classes
        self.k_shot = k_shot
        self.info_temperature = info_temperature
        self.feature_dim = feature_dim
        self.labeled_size = self.num_classes * self.k_shot
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

        if self.labeled_size > 0:
            labeled_features = torch.rand(self.labeled_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
            self.register_buffer('labeled_features', labeled_features)

    def forward(self, q, k, k_all, buffer_idx):
        k = k[:self.batch_size].detach()

        l_pos = (q * k).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        # TODO: remove clone. need update memory in backwards
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()

        if self.labeled_size > 0:
            # update memory
            with torch.no_grad():
                k_all = k_all.reshape(dist.get_world_size(), -1, self.feature_dim)
                k_all_unlabeled = k_all[:, :self.batch_size, :].reshape(-1, self.feature_dim)
                k_all_labeled = k_all[:, self.batch_size:, :].reshape(-1, self.feature_dim)
                all_size = k_all_unlabeled.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
                self.memory.index_copy_(0, out_ids, k_all_unlabeled)
                self.index = (self.index + all_size) % self.queue_size

            out2 = torch.mm(q, self.labeled_features.clone().detach().t()).div(self.info_temperature).softmax(-1).view(q.size(0), self.num_classes, self.k_shot).sum(2)
            buffer_idx_all = dist_collect(buffer_idx)
            with torch.no_grad():
                self.labeled_features.index_copy_(0, buffer_idx_all, k_all_labeled)
            return out, out2
        else:
            # update memory
            with torch.no_grad():
                all_size = k_all.shape[0]
                out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
                self.memory.index_copy_(0, out_ids, k_all)
                self.index = (self.index + all_size) % self.queue_size
            return out
