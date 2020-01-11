import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.distributions import Normal

# Initialization of Graph Input
edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=torch.long)
X = torch.randn(5,100)
data = Data(x=X,edge_index=edge_index.t().contiguous())

# Definition of Encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder1 = GATConv(100,50)
        self.encoder2 = GATConv(50,4)

    def forward(self,x):
        x, edge_index = x.x, x.edge_index
        x = F.relu(self.encoder1(x,edge_index))
        x = F.relu(self.encoder2(x,edge_index))
        return x, edge_index

# Definition of Encoder
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder,self).__init__()
        self.decoder1 = GATConv(4,50)
        self.decoder2 = GATConv(50,100)

    def forward(self,x,edge_index):
        x = F.relu(self.decoder1(x,edge_index))
        x = F.relu(self.decoder2(x,edge_index))
        return x, edge_index

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
        z1,z2,s1 = self.inverse_a(z1,z2,edge_index)
        z2,z1,s2 = self.inverse_b(z2,z1,edge_index)
        # Calculate Jacobian
        log_det_zx = torch.sum(-s1,axis=1) + torch.sum(-s2,axis=1)
        return z1, z2, log_det_zx

encoder = Encoder()
decoder = Decoder()
gnf = GNF()
max_epoch = 1 

for epoch in range(max_epoch):
    data,edge_index = encoder(data)
    # Forward
    z1,z2,log_det_xz = gnf.forward(data,edge_index)
    log_det_xz = log_det_xz.unsqueeze(0)
    log_det_xz = log_det_xz.mm(torch.ones(log_det_xz.t().size()))
    #self.log_det_xz.append(log_det_xz)
    # Sample
    z = torch.cat((z1,z2),dim=1)
    n = Normal(z, torch.randn(z.size()))
    z = n.sample()
    # Inverse
    z1,z2, log_det_zx = gnf.inverse(z,edge_index)
    log_det_zx = log_det_zx.unsqueeze(0)
    log_det_zx = log_det_zx.mm(torch.ones(log_det_zx.t().size()))
    # Decoder
    z = torch.cat((z1,z2),dim=1)
    y = decoder(z,edge_index)

print('Done')
