import torch
from torch import nn
import torch.nn.functional as F

from typing import List

import torch
from torch import nn, Tensor, sparse
from torch.nn import functional as F, init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias_start: float = 0.0):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        self.bias = nn.Parameter(Tensor(out_features))

        init.xavier_normal_(self.weight, gain=1.414)
        init.constant_(self.bias, val=bias_start)

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape
        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = x0.unsqueeze(dim=0)  # (1, num_nodes, input_dim * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)


class ChebNet(nn.Module):
    def __init__(self, f_in: int, f_out: int, n_matrices: int):
        super(ChebNet, self).__init__()
        self.out = nn.Linear(n_matrices * f_in, f_out)

    def forward(self, signals: Tensor, supports: Tensor) -> Tensor:
        """
        implement of ChebNet
        :param signals: input signals, Tensor, [*, N, F_in]
        :param supports: pre-calculated Chebychev polynomial filters, Tensor, [N, n_matrices, N]
        :return: Tensor, [B, N, F_out]
        """
        # shape => [B, N, K, F_in]
        tmp = supports.matmul(signals.unsqueeze(-3))
        # shape => [B, N, F_out]
        return self.out(tmp.reshape(tmp.shape[:-2] + (-1,)))


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, inputs):
        """
        :param inputs: tensor, [N, C_{in}, L_{in}]
        :return: tensor, [N, C_{out}, L_{out}]
        """
        outputs = super(CausalConv1d, self).forward(F.pad(inputs, [self.__padding, 0]))
        return outputs[:, :, :outputs.shape[-1] - self.__padding]


class Edgewise_GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, graph_weight_dim):
        super(Edgewise_GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.att_fc = nn.Linear(out_dim*2+graph_weight_dim, 1, bias=True)

    def message_pass(self, edges):
        src = edges.src['z']           #b,f
        dst = edges.dst['z']         #b,f
        weight = edges.data['weight']
        batch = src.shape[1]
        weight = weight.repeat(1,batch).unsqueeze(2)
        e = F.leaky_relu(self.att_fc(torch.cat([src, dst, weight], dim=2)))
        return {'z':edges.src['z'], 'e':e}

    def message_reduce(self, nodes):
        alpha = F.softmax(nodes.mailbox['z'], dim=1)           #???????????不确定维度
        h = torch.sum(alpha * nodes.mailbox['e'], dim=1)
        return {'h':h}

    def forward(self,h):             #b,n,f
        z = self.fc(h)             #b,n,output_num
        z = z.transpose(0,1)        #n,b,f
        g = self.g.local_var()
        g.ndata['z'] = z
        g.update_all(self.message_pass,self.message_reduce)
        return g.ndata.pop('h').transpose(0,1)            #b,n,f ?????


class Multihead_EW_GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, graph_weight_dim, num_head, merge='mean'):
        super(Multihead_EW_GATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(Edgewise_GATLayer(g, in_dim, out_dim,graph_weight_dim))
        self.merge = merge

    def forward(self, x):                  #b,n,f
        head_outs = [att_head(x) for att_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)