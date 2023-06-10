import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.distributions import Normal
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
from torch.optim import Adam
from torch_geometric.utils import train_test_split_edges


# Definition of Encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder1 = GATConv(1433,50)
        self.encoder2 = GATConv(50,4)

    def forward(self,x):
        x, edge_index = x.x, x.train_pos_edge_index
        x = F.relu(self.encoder1(x,edge_index))
        x = F.relu(self.encoder2(x,edge_index))
        return x, edge_index

# Definition of Encoder
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder,self).__init__()
        self.decoder1 = GATConv(4,50)
        self.decoder2 = GATConv(50,1433)

    def forward(self,x,edge_index):
        x = F.relu(self.decoder1(x,edge_index))
        x = F.relu(self.decoder2(x,edge_index))
        adj_pred = InnerProductDecoder().forward_all(x, sigmoid= True)
        return adj_pred, edge_index

# Definition of GNF
class GNF(nn.Module):

    def __init__(self):
        super(GNF,self).__init__()
        self.F1 = GATConv(2,2)
        self.F2 = GATConv(2,2)
        self.G1 = GATConv(2,2)
        self.G2 = GATConv(2,2)

    def forward(self,x,edge_index):
        x1,x2 = x[:,:2], x[:,2:]
        x1,x2,s1 = self.f_a(x1,x2,edge_index)
        x2,x1,s2 = self.f_b(x2,x1,edge_index)
        # Calculate Jacobian
        log_det_xz = torch.sum(s1,axis=1) + torch.sum(s2,axis=1)
        return x1, x2, log_det_xz

    def f_a(self,x1,x2,edge_index):
        s_x = self.F1(x1,edge_index)
        t_x = self.F2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def f_b(self,x1,x2,edge_index):
        s_x = self.G1(x1,edge_index)
        t_x = self.G2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def inverse_b(self,z1,z2,edge_index):
        s_x = self.G1(z1,edge_index)
        t_x = self.G2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse_a(self,z1,z2,edge_index):
        s_x = self.F1(z1,edge_index)
        t_x = self.F2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse(self,z,edge_index):
        z1,z2 = z[:,:2], z[:,2:]
        z1,z2,s1 = self.inverse_b(z1,z2,edge_index)
        z2,z1,s2 = self.inverse_a(z2,z1,edge_index)
        # Calculate Jacobian
        log_det_zx = torch.sum(-s1,axis=1) + torch.sum(-s2,axis=1)
        return z1, z2, log_det_zx



if __name__ == "__main__":

    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid("datasets", 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    all_edge_index = data.edge_index
    data = train_test_split_edges(data, 0.02, 0.01)
        

    encoder = Encoder()
    decoder = Decoder()
    gnf = GNF()

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(gnf.parameters())
    optimizer = Adam(params , lr=0.01)
    max_epoch = 10



    for epoch in range(max_epoch):

        data_, edge_index = encoder(data)
        # Forward

        z1,z2,log_det_xz = gnf.forward(data_,edge_index)

        log_det_xz = log_det_xz.unsqueeze(0)
        log_det_xz = log_det_xz.mm(torch.ones(log_det_xz.t().size()))
        #self.log_det_xz.append(log_det_xz)



        # Sample
        z = torch.cat((z1,z2),dim=1)
        n = Normal(z, torch.ones(z.size()))
        z = n.sample()
        # Inverse
        z1,z2, log_det_zx = gnf.inverse(z,edge_index)
        log_det_zx = log_det_zx.unsqueeze(0)
        log_det_zx = log_det_zx.mm(torch.ones(log_det_zx.t().size()))
        # Decoder
        z = torch.cat((z1,z2),dim=1)
        y, _ = decoder(z,edge_index)

        pos_loss = -torch.log(
            y).mean() 
        

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), data.train_pos_edge_index.size(1))

        neg_loss = -torch.log(1 - y + 1e-15).mean()

        loss = neg_loss + pos_loss
        loss.backward()
        optimizer.step()

  

    print('Done')
