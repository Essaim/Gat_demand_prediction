from typing import Tuple

import torch
from torch import nn, Tensor
import math
import random
import numpy as np

class LinearDecompose(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 hidden_layer1,
                 hidden_layer2,
                 input_dim,
                 output_dim,
                 node_num):
        super(LinearDecompose, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim * node_num, hidden_layer1),
                                     nn.Linear(hidden_layer1, hidden_layer2),
                                     nn.Linear(hidden_layer2, hidden_size))
        self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_layer2),
                                     nn.Linear(hidden_layer2, hidden_layer1),
                                     nn.Linear(hidden_layer1, output_dim * node_num))

    def forward(self,x):
        """
        Linear Decompose
        :param inputs: [N,B, input_dim]
        :return: [N,B, input_dim]
        """
        b, n, f = x.shape
        x = x.reshape(b, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(b, n ,f)
        return x

class SvdDecompose():
    def __init__(self,
                 hidden_size,
                 loss):
        self.hidden_size = hidden_size
        self.w = None
        self.wt = None
        self.loss_func = loss


    def decompose(self, inputs):
        """
        SVD Decompose
        :param inputs: [N,B , input_dim]
        :return: []
        """
        T, N, input_dim = inputs.shape
        inputs = inputs.reshape(T, -1)
        u, s, v = torch.svd(inputs, True)
        self.w = torch.diag(s[:self.hidden_size]).mm(v[:,:self.hidden_size].T)
        loss = self.loss_func(u[:, :self.hidden_size].mm(self.w), inputs)
        print(loss.item())
        self.wt = torch.pinverse(self.w)

    def encoder(self, x):
        """
        SVD Decompose
        :param inputs: [N, B , input_dim]
        :return:[N, B, svd_dim]
        """
        N, B ,input_dim = x.shape
        x = x.transpose(0,1)        #  B,N,input_dim
        wt = self.wt.repeat(B, 1, 1)
        x_encode = x.bmm(wt)         #B,N,hidden_size
        return x_encode.transpose(1,0)
    def decoder(self, x):
        """
        SVD Decompose
        :param inputs: [b,n , hidden_size]
        :return: [b,n , output_size]
        """
        B, N, hidden_size = x.shape
        w = self.w.repeat(B,1,1)
        x_decode = x.bmm(w)
        return x_decode


class Costnet(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 n_hist: int,
                 n_pred: int,
                 hidden_size: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 node_num: int,
                 decay_steps):
        super(Costnet, self).__init__()
        self.encoder_linear = encoder
        self.decoder_linear = decoder
        for p in self.parameters():
            p.requires_grad = False
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.output_dim = output_dim
        self.encoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)
        self.cl_decay_steps = decay_steps


    def forward(self, inputs: Tensor, targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, [B, n_pred, N, input_dim]
        """
        # print(inputs.shape,targets.shape)
        b, n_hist, n, input_dim = inputs.shape
        inputs = inputs.transpose(0, 1).reshape(self.n_hist, b, -1)
        inputs = self.encoder_linear(inputs)

        if targets is not None:
            targets = self.encoder_linear(targets.transpose(0, 1).reshape(self.n_pred, b, -1))
        h, c = self.encoding(inputs)
        outputs = self.decoding((h, c), targets, self._compute_sampling_threshold(batch_seen))
        outputs = outputs.transpose(0, 1)         #b, n_pred, hidden_size
        return self.decoder_linear(outputs).reshape(b, self.n_pred, n, -1)

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

    def encoding(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        encoding
        :param inputs: tensor, [n_hist, B * N, input_dim]
        :return: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        """
        _, (h, c) = self.encoder(inputs)
        return h, c

    def decoding(self, hc: Tuple[Tensor, Tensor], targets: Tensor, teacher_force: bool = 0.5):
        """
        decoding
        :param hc: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        :param targets: optional, exists while training, tensor, [n_pred, B, N, output_dim]
        :return: tensor, shape as same as targets
        """
        h, c = hc
        decoder_input = torch.zeros(1, h.shape[1], h.shape[2], device=h.device, dtype=h.dtype)

        outputs = list()
        for t in range(self.n_pred):
            decoder_input, (h, c) = self.decoder(decoder_input, (h, c))
            # decoder_input = self.projector(decoder_input)
            outputs.append(decoder_input)
            if targets is not None and random.random() < teacher_force:
                decoder_input = targets[t].unsqueeze(0)
        return torch.cat(outputs)
