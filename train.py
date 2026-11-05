import argparse
import os
import os.path as osp
import pytorch_lightning as pl
import lightning_lite
import torch.nn.functional as F
from utils import utils
from data.data import get_dl, gety
from models.GTCN import GTCN
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
import wandb


parser = argparse.ArgumentParser(description='Traing GTCN')
parser.add_argument('--gpu', '-g', dest='gpu', default=0)
parser.add_argument('--seed', '-s', dest='seed', default=0, type=int)
parser.add_argument('--fold', '-f', dest='fold', default=0, type=int)
args = parser.parse_args()

gpu_ID = str()
a = list(range(int(args.gpu)))
for s in a:
    gpu_ID += str(s) + ','
gpu_ID = gpu_ID[:-1]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ID

lightning_lite.utilities.seed.seed_everything(seed=args.seed, workers=True)

yaml_path = osp.join(os.getcwd(), 'pt_v20250328.yaml')
config = utils.load_yaml(yaml_path)

model = GTCN(config)

wandb_logger = WandbLogger(project='GTCN_gapv1', name='gap2', )
wandb_logger.watch(model, log='all', log_freq=10)

def train(config, data_path, Total_num, splitpara, logger=True):

    print(config)

    train_dl, valid_dl = get_dl(config, data_path, Total_num, splitpara)
    print(f'train: {len(train_dl)} valid: {len(valid_dl)}')

    print('Start training:')
    EPOCHS = config.epochs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_mae', mode='min')
    pcb = TQDMProgressBar(refresh_rate=4)
    trainer = pl.Trainer(accelerator='ddp', devices=8, num_nodes = 2,
                     strategy='ddp',
                     max_epochs=EPOCHS, 
                     callbacks=[checkpoint_callback, pcb],
                     logger=wandb_logger,
                     precision=16,
                     gradient_clip_val=config.gradient_clip_val,
                     gradient_clip_algorithm="value"
                    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    print('Predict valid:')
    y_p = trainer.predict(model, valid_dl)
    y_p = torch.cat(y_p, dim=0)
    y = gety(valid_dl)

    score1 = F.l1_loss(y_p, y).item()

    msgs = {'VALID MAE':score1}
    print(f'Valid MAE: {score1:.4f}')
    print("weights saved:", trainer.log_dir)
    return msgs


if __name__ == "__main__":
    quick_run = False                                   
    
    dataset_path = '/home/xhwan/data_processed'
    train(config=config, data_path=dataset_path, Total_num=50000, splitpara=[40000, 10000])
    
"""     config = utils.load_yaml(yaml_path)
    train_dl, valid_dl = get_dl(config, data_path=dataset_path, Total_num=31339, splitpara=[25000, 6339])
    model = GTCN(config)
    for batch in train_dl:
        x_out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        print(x_out)
        loss = F.smooth_l1_loss(x_out, batch.y2, beta=0.1)
        print(loss) """

    #print(batch.batch)
    #print(batch.x)
    #print(batch.edge_index)
    #print(batch.edge_attr)
    #print(batch.y1)
    #print(batch.y2)
    
    #x_out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    #print(x_out)
    #loss = F.smooth_l1_loss(x_out, batch.y2, beta=0.1)
    #print(loss)
