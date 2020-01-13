import torch
from torch import nn
import torch.nn.functional as F
from .basemodel import Multihead_EW_GATLayer


class LinearAttention(nn.Module):
    def __init__(self, in_size, graph_num, hidden_size=128):
        super(LinearAttention, self).__init__()
        self.project = nn.ModuleList()
        for _ in range(graph_num):
            self.project.append(nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            ))
        self.graph_num = graph_num

    def forward(self, z):                        #graph_num, b,n,f
        project_out = []
        for _z,pro in zip(z,self.project):
            project_out.append(pro(_z))
        project_out = torch.stack(project_out, dim=0)
        beta = torch.softmax(project_out, dim=0)
        graph_wise_out = torch.sum(beta * z, dim=0)
        return graph_wise_out


class H_GATcell(nn.Module):
    def __init__(self, graph, in_dim, out_dim, graph_weight_dim, num_head, merge='mean'):
        super(H_GATcell, self).__init__()
        self.graph = graph
        self.graph_num = len(graph)
        self.graph_wise_gat = nn.ModuleList()
        for i in range(self.graph_num):
            self.graph_wise_gat.append(
                Multihead_EW_GATLayer(graph[i], in_dim, out_dim, graph_weight_dim, num_head, merge='mean'))
        self.linearatt = LinearAttention(in_dim, self.graph_num)  # ??????????
        # self.alfa = nn.ParameterList()
        # for i in graph_cnt:x
        #     self.alfa.append(torch.zeros(1))
        # self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):                           #b,n,f
        graph_wise_gat_out = []
        for EW_GATLaer in self.graph_wise_gat:
            graph_wise_gat_out.append(EW_GATLaer(x))
        H_gat_out = torch.stack(graph_wise_gat_out, dim=0)  # graph_num, b,n,f
        latt_out = self.linearatt(H_gat_out)
        # latt_out.append(self.linear(x))
        # latt_out = torch.stack(latt_out, dim=1)

        return latt_out


class HGAT_LSTM(nn.Module):
    def __init__(self, graph, layer_num, node_num, concate_num, predict_dim, in_dim, out_dim, graph_weight_dim,
                 num_head, merge='mean'):
        super(HGAT_LSTM, self).__init__()
        self.bike_graph = graph[0]
        self.taxi_graph = graph[1]
        self.HGAT_bike = nn.ModuleList()
        self.HGAT_taxi = nn.ModuleList()
        self.concate_num = concate_num
        for _ in range(layer_num):
            self.HGAT_bike.append(H_GATcell(self.bike_graph, in_dim, out_dim, graph_weight_dim, num_head, merge))
            self.HGAT_taxi.append(H_GATcell(self.taxi_graph, in_dim, out_dim, graph_weight_dim, num_head, merge))
        self.lstm = nn.LSTM(input_size=node_num * out_dim,
                            hidden_size=node_num * out_dim,
                            num_layers=1,
                            batch_first=True)
        self.node_num = node_num
        self.predict_dim = predict_dim

    def concate(self, bike, taxi):
        bike, taxi = bike.transpose(0, 1), taxi.transpose(0, 1)
        x = torch.cat([bike[:self.concate_num], taxi[self.concate_num:]], dim=0)
        return x.transpose(0, 1)

    def forward(self, x):

        x = x.transpose(0, 1)         #t,b,n,f
        HGAT_out = list()
        # i = 1
        for each in x:
            # print(i)
            # i +=1
            h = each                #b,n,f
            for hgat_bike, hgat_taxi in zip(self.HGAT_bike, self.HGAT_taxi):
                bike_h = hgat_bike(h)             #b,n,f
                taxi_h = hgat_taxi(h)
                h = self.concate(bike_h, taxi_h)  # b , n ,f
            HGAT_out.append(h)
        HGAT_out = torch.stack(HGAT_out, dim=0).transpose(0, 1)  # b,t ,n f
        LSTM_in = HGAT_out.view(HGAT_out.shape[0], HGAT_out.shape[1], -1)     #b,t,n*f
        output, (h_n, c_n) = self.lstm(LSTM_in)
        output_last_timestep = h_n[-1, :, :]  # b,n*f
        out = output_last_timestep.view(-1,1, self.node_num, self.predict_dim)  # b, n, f
        # print(i)
        return out
