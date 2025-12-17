import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, global_mean_pool, DeepGCNLayer
from models.GTCNlayer import GTCNlayer
from torch.nn import Dropout, Linear, ReLU, LayerNorm
import torch.nn.functional as F
import torch


class GTCN(pl.LightningModule):

    def __init__(self, config):
        super(GTCN, self).__init__()

        self.Node_fea = config.nd_fea
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs
        self.YMIN = config.Gap_min
        self.YMAX = config.Gap_max
        # hidden layer node features
        self.hidden = config.hnrons
        self.En = config.E_node
        Ee = config.E_edge
        self.max_node_fea = config.max_fea_val + 1
        self.emb = torch.nn.Embedding(self.max_node_fea, self.En)
        heads = config.heads
        h_out = self.hidden // heads

        act = None if config.act == 'None' else config.act
        act = eval(config.act)
        self.layer_begin = GTCNlayer(self.Node_fea*self.En, h_out, heads, edge_dim=Ee, act=act, fill_value=1.0, beta=config.beta)
        Layers = []
        for i in range(config.layers):
            Layers.append(DeepGCNLayer(GTCNlayer(h_out*heads, h_out, heads, edge_dim=Ee, act=act, fill_value=1.0, beta=config.beta)))
        self.layers = torch.nn.ModuleList(Layers)
        self.w = torch.nn.Sequential(Linear(h_out*heads, 4), ReLU(), Linear(4, 1))
        self.out = torch.nn.Sequential(Linear(h_out*heads, 1),
                                       Linear(1, 1))      

    def forward(self, x, edge_index, edge_attr, batch_index):
        edge_attr = edge_attr.float()/4.0
        #print('EA1:', edge_attr)
        edge_attr[:, 2]= edge_attr[:, 2]/10
        #print('EA2:', edge_attr)
        x = self.emb(x).view(-1, self.Node_fea*self.En)
        x = self.layer_begin(x, edge_index, edge_attr)
        x = F.relu(x)
        for i, l in enumerate(self.layers):
            x = l(x, edge_index, edge_attr)
            x = F.relu(x)
        #x = global_mean_pool(x, batch_index)
        #print('x', x)
        a = self.w(x)
        #print('a_matirx', a)
        a = torch.exp(a)
        x = global_add_pool(x*a, batch_index)
        a = global_add_pool(a, batch_index)
        x = x/a
        x1 = self.out(x)
        x2 = torch.clip(x1, self.YMIN, self.YMAX)
        x_out = (x1+x2)/2
        return x_out.squeeze()
    
    def _f(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = batch.edge_attr
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, edge_attr, batch_index)
        return x_out
    
    def _loss(self, batch, batch_index, tag):
        x_out = self._f(batch, batch_index)
        loss = F.smooth_l1_loss(x_out, batch.y2, beta=0.1)
        x_out = torch.clip(x_out, self.YMIN, self.YMAX)
        mae = F.l1_loss(x_out, batch.y2)
        self.log(f"{tag}_mae", mae, batch_size = batch.y2.shape[0], prog_bar=True)
        print(f"{tag}", batch.y2.shape[0])
        return loss

    def training_step(self, batch, batch_index):
        return self._loss(batch, batch_index, 'train')

    def validation_step(self, batch, batch_index):
        return self._loss(batch, batch_index, 'valid')
        
    def predict_step(self, batch, batch_index):
        x_out = self._f(batch, batch_index)
        return torch.clip(x_out, self.YMIN, self.YMAX)

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        slr = torch.optim.lr_scheduler.CosineAnnealingLR(adam, self.epochs)
        return [adam], [slr]





if __name__ == "__main__":
    print('GTCN loaded')
