from typing import Tuple

import torch
from torch import nn, Tensor
import math
import random


class FCLSTM(nn.Module):
    def __init__(self,
                 n_hist: int,
                 n_pred: int,
                 hidden_size: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 decay_steps: int):
        super(FCLSTM, self).__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.output_dim = output_dim
        self.encoder = nn.LSTM(input_dim, hidden_size, n_rnn_layers)
        self.decoder = nn.LSTM(output_dim, hidden_size, n_rnn_layers)
        self.projector = nn.Linear(hidden_size, output_dim)
        self.cl_decay_steps = decay_steps

    def forward(self, inputs: Tensor, targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, [B, n_pred, N, input_dim]
        """
        b, _, n, input_dim = inputs.shape
        inputs = inputs.transpose(0, 1).reshape(self.n_hist, b * n, -1)
        if targets is not None:
            targets = targets.transpose(0, 1).reshape(self.n_pred, b * n, -1)
        h, c = self.encoding(inputs)
        outputs = self.decoding((h, c), targets, self._compute_sampling_threshold(batch_seen))
        return outputs.reshape(self.n_pred, b, n, -1).transpose(0, 1)

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
        decoder_input = torch.zeros(1, h.shape[1], self.output_dim, device=h.device, dtype=h.dtype)

        outputs = list()
        for t in range(self.n_pred):
            decoder_input, (h, c) = self.decoder(decoder_input, (h, c))
            decoder_input = self.projector(decoder_input)
            outputs.append(decoder_input)
            if targets is not None and random.random() < teacher_force:
                decoder_input = targets[t].unsqueeze(0)
        return torch.cat(outputs)


class FCLSTM_BNode(nn.Module):
    def __init__(self,
                 n_hist: int,
                 n_pred: int,
                 hidden_size: int,
                 hidden_layer1,
                 hidden_layer2,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 node_num: int,
                 decay_steps):
        super(FCLSTM_BNode, self).__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.output_dim = output_dim
        self.encoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)
        self.projector_encode = nn.Sequential(nn.Linear(input_dim * node_num, hidden_layer1),
                                              nn.Linear(hidden_layer1, hidden_layer2),
                                              nn.Linear(hidden_layer2, hidden_size))
        self.projector_decode = nn.Sequential(nn.Linear(hidden_size, hidden_layer2),
                                              nn.Linear(hidden_layer2, hidden_layer1),
                                              nn.Linear(hidden_layer1, input_dim * node_num))
        self.cl_decay_steps = decay_steps

    def forward(self, inputs: Tensor, targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, [B, n_pred, N, input_dim]
        """
        b, _, n, input_dim = inputs.shape
        inputs = inputs.transpose(0, 1).reshape(self.n_hist, b, -1)
        inputs = self.projector_encode(inputs)
        if targets is not None:
            targets = self.projector_encode(targets.transpose(0, 1).reshape(self.n_pred, b, -1))
        h, c = self.encoding(inputs)
        outputs = self.decoding((h, c), targets, self._compute_sampling_threshold(batch_seen))
        outputs = outputs.transpose(0, 1)
        return self.projector_decode(outputs).reshape(b, self.n_pred, n, -1)

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
        # return 0

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
