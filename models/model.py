from typing import Tuple, List

from .h_gat import HGAT_LSTM
import numpy as np
import dgl
from dgl import DGLGraph, init
from scipy import sparse
from .dcrnn import DCRNN
from .fc_lstm import FCLSTM, FCLSTM_BNode
from .graph_wavenet import GWNet
from .stg2seq import STG2Seq
from .costnet import LinearDecompose, Costnet, SvdDecompose
import torch
from torch import nn, Tensor
from utils.graph import sparse_scipy2torch, load_graph_data


def graph_preprocess(model_name, graph_h5, data_category, device):
    if model_name == 'H_GAT':
        dis_bb = graph_h5['dis_bb'][:]
        dis_bt = graph_h5['dis_bt'][:]
        trans_bb = graph_h5['trans_bb'][:]
        dis_tb = graph_h5['dis_tb'][:]
        dis_tt = graph_h5['dis_tt'][:]
        trans_tt = graph_h5['trans_tt'][:]
        bike_num = dis_bb.shape[0]
        taxi_num = dis_tt.shape[0]

        print(bike_num, taxi_num)
        graph = []
        for i in [dis_bb, dis_bt, trans_bb, dis_tt, dis_tb, trans_tt]:
            g = DGLGraph()
            g.add_nodes(bike_num + taxi_num)
            matrix = sparse.coo_matrix(i)
            src = matrix.row if len(matrix.row) == bike_num else matrix.row + bike_num
            dis = matrix.col if len(matrix.col) == bike_num else matrix.col + bike_num
            wei = torch.tensor(matrix.data).unsqueeze(1).float().to(device)
            # print(wei.shape)
            g.add_edges(src, dis, {'weight': wei})
            # print(g.number_of_edges())
            graph.append(g)
            # x = input()
        bike_graph = graph[:len(graph) // 2]
        taxi_graph = graph[len(graph) // 2:]

        return [bike_graph, taxi_graph]
    elif model_name == 'DCRNN':
        if data_category[0] == 'taxi':
            matrix = graph_h5['dis_tt'][:]
        elif data_category[0] == 'bike':
            matrix = graph_h5['dis_bb'][:]
        matrix = sparse.coo_matrix(matrix)
        return [matrix]
    elif model_name == 'STG2Seq':
        if data_category[0] == 'taxi':
            matrix = graph_h5['dis_tt'][:]
        elif data_category[0] == 'bike':
            matrix = graph_h5['dis_bb'][:]
        matrix = sparse.coo_matrix(matrix)
        return matrix





def create_model(model_name, loss, conf, data_category, device, graph_h5, encoder=None, decoder=None):
    graph = graph_preprocess(model_name, graph_h5, data_category, device)
    # graph = None
    graph_h5.close()
    if model_name == 'H_GAT':
        model = HGAT_LSTM(graph, **conf)
        return model, None
    elif model_name == 'DCRNN':
        model = DCRNN(**conf)
        return model, DCRNNTrainer(model, loss, [sparse_scipy2torch(g).to(device) for g in graph])
    elif model_name == 'FCLSTM':
        model = FCLSTM(**conf)
        return model, FCLSTMTrainer(model, loss)
    elif model_name == 'FCLSTM_BNode':
        model = FCLSTM_BNode(**conf)
        return model, FCLSTMTrainer(model, loss)
    elif model_name == 'LinearDecompose':
        model = LinearDecompose(**conf)
        return model, DecomposeTrainer(model, loss)
    elif model_name == 'SvdDecompose':
        model = SvdDecompose(**conf, loss=loss)
        return model, None
    elif model_name == 'Costnet':
        model = Costnet(**conf, encoder=encoder, decoder=decoder)
        return model, CostnetTrainer(model, loss)
    elif model_name == 'STG2Seq':
        if graph is not None:
            model = STG2Seq(**conf, graph=sparse_scipy2torch(graph).to(device),device=device)
        else:
            model = STG2Seq(**conf, graph=None,device=device)
        return model, STG2SeqTrainer(model, loss)




    elif model_name == 'GWNET':
        adjtype, randomadj = conf.pop('adjtype'), conf.pop('randomadj')
        if adjtype is not None:
            supports = [torch.tensor(graph.todense(), dtype=torch.float32, device=device) for graph in
                        load_graph_data(dataset, adjtype)]
            aptinit = None if randomadj else supports[0]
        else:
            supports = aptinit = None
        model = GWNet(device, supports=supports, aptinit=aptinit, **conf)
        return model, GWNetTrainer(model, loss)


class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class DCRNNTrainer(Trainer):
    def __init__(self, model: DCRNN, loss, graphs: List[Tensor]):
        super(DCRNNTrainer, self).__init__(model, loss)
        for graph in graphs:
            graph.requires_grad_(False)
        self.graphs = graphs
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, self.graphs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss


class FCLSTMTrainer(Trainer):
    def __init__(self, model, loss):
        super(FCLSTMTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss


class DecomposeTrainer(Trainer):
    def train(self, input, targets, phase):
        outputs = self.model(input)
        loss = self.loss(outputs, targets)
        return outputs, loss

    def get_loss(self):
        return self.loss


class CostnetTrainer(Trainer):
    def __init__(self, model, loss):
        super(CostnetTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0

    def train(self, inputs, targets, phase):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss

class STG2SeqTrainer(Trainer):
    def __init__(self, model, loss):
        super(STG2SeqTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0
    def train(self, inputs, targets, phase):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss

class GWNetTrainer(Trainer):
    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        outputs = self.model(inputs.transpose(1, 3))
        loss = self.loss(outputs, targets)
        return outputs, loss