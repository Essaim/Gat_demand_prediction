import torch
import numpy
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import sparse
import math


class STG2Seq(nn.Module):
    def __init__(self, graph, node_num, input_dim, output_dim, hidden_dim,adaptive_dim, n_hist, n_pred, patch_size, sliding_window,
                 shortterm_len, longterm_len, cl_decay_steps, device):
        super(STG2Seq, self).__init__()
        self.graph = graph
        # if self.graph is None:
        self.nodevec1 = nn.Parameter(torch.randn(node_num, adaptive_dim), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(adaptive_dim, node_num), requires_grad=True).to(device)
            # self.graph = self.nodevec1.mm(self.nodevec2).to_sparse()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.sliding_window = sliding_window
        self.long_term = GGCM(longterm_len, input_dim, output_dim, hidden_dim, patch_size)
        self.short_term = GGCM(shortterm_len, input_dim, output_dim, hidden_dim, patch_size)
        self.attention = Attention(input_dim)
        self.cl_decay_steps = cl_decay_steps

    def forward(self, x, targets, batch_seen):
        """
        STG2Seq convolutional recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param inputs: [B, n_pred, N, input_dim]
        :return: [B, n_pred, N, output_dim]
        """
        graph = self.nodevec1.mm(self.nodevec2)
        graph += self.graph.to_dense()
        graph = graph.to_sparse()

        b, n_hist, n, input_dim = x.shape
        x = x.transpose(0, 1)  # n_hist, b, n, input_dim
        longterm_out = self.long_term(x, graph)  # n_hist, b, n, input_dim
        y = []
        for i in range(self.n_pred):
            shortterm_out = self.short_term(x[-self.sliding_window:], graph)  # short_term_len, b, n, input_dim
            out = self.attention(torch.cat([longterm_out, shortterm_out], dim=0))  # b, n, input_dim
            y.append(out)
            if targets is not None and random.random() < self._compute_sampling_threshold(batch_seen):
                out = targets.transpose(0, 1)[i]
            x = torch.cat([x, out.unsqueeze(0)], dim=0)  # n_hist++, b, n, input_dim
        y = torch.stack(y, dim=1)  # b,n_pred, n, input_dim
        return y

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
        # return 0


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1, bias=True)

    def forward(self, x):
        """
        attention layer
        :param inputs: [t, b, n, input_dim]
        :return: [t, b, N, output_dim]
        """
        # print(x.transpose(2, 3)[0][0][0].cpu().detach().numpy().tolist())
        # print(11111)
        alfa = torch.softmax(torch.tanh(self.linear(x)), dim=0)  # [t, b, N, 1]
        # print(alfa.transpose(2, 3)[0][0][0].cpu().detach().numpy().tolist())
        # print(11111)
        out = torch.sum(alfa * x, dim=0)  # [b, N, output_dim]
        return out


class GGCM(nn.ModuleList):
    def __init__(self, len, input_dim, output_dim, hidden_dim, patch_size):
        super(GGCM, self).__init__()
        self.append(GGCMCell(input_dim, hidden_dim, patch_size))
        for _ in range(len - 2):
            self.append(GGCMCell(hidden_dim, hidden_dim, patch_size))
        self.append(GGCMCell(hidden_dim, output_dim, patch_size))

    def forward(self, x, supports):
        out = x  # t,b,n,f
        for Cell in self:
            out = Cell(out, supports)  # t,b,n,f
        return out


class GGCMCell(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super(GGCMCell, self).__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = STGGraphConv(input_dim, output_dim, patch_size)

    def forward(self, x, supports):
        """
       GGCM
       :param inputs: [B, n_hist, N, input_dim]
       :param supports: list of tensors, each tensor is with shape [N, N]
       :return: [B, n_pred, N, output_dim]
       """
        n_hist, b, n, input_dim = x.shape  # t, b, n, f
        # x = x.transpose(0, 1)
        padding = torch.zeros(self.patch_size - 1, b, n, self.input_dim, device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=0)
        out = []
        for i in range(n_hist):
            y = self.conv(x[i:i + self.patch_size], supports)  # in: pathch_size,b,n,f    out:b,n,f
            out.append(y)
        out = torch.stack(out, dim=0)  # t,b,n,f
        return out


class STGGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super(STGGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.linear = nn.Linear(input_dim * patch_size, output_dim * 2)
        self.activate = nn.Sigmoid()
        self.linear2 = nn.Linear(input_dim * patch_size, output_dim)

    def _concat(self, x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, x, supports):
        """
        GGCM
        :param inputs: [pathch_size,batch_size,N,input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :return: [batch_size,N,output_dim]
        """
        inputs = x
        p, b, n, input_dim = x.shape
        x = x.permute([2, 1, 0, 3]).reshape(n, -1)  # N, batch, patch_szie, input_dim  -->  N, -1
        # x = x0.unsqueeze(0)
        #
        # for support in supports:
        #     x1 = support.mm(x0)
        #     x = self._concat(x, x1)
        #     for k in range(2, self.max_diffusion_step+1):
        #         x2 = 2 * support.mm(x1) - x0
        #         x = self._concat(x, x2)
        #         x1, x0 = x2, x1
        #
        # x = x.view(-1, n, b, p * input_dim).permute([2,1,0,])

        x = sparse.mm(supports, x)
        x = x.reshape(n, b, p * input_dim).transpose(0, 1)  # batch,N,p*input_dim
        linear_out = self.linear(x)
        x, gate = linear_out.chunk(2, 2)

        inputs = inputs.permute([1, 2, 0, 3]).reshape(b, n, -1)  # batch, N,p*input_dim
        inputs = self.linear2(inputs)
        return (x + inputs) * self.activate(gate)  # batch,N,output_dim
