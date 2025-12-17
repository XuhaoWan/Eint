import argparse
import os
import os.path as osp
import pytorch_lightning as pl
import lightning_lite
import torch
import torch.nn.functional as F

from utils import utils
from data.data import get_dl, gety
from models.GTCN import GTCN_ft
from models.GTCN_ft import GTCNFinetune

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb


parser = argparse.ArgumentParser(description='Fine-tune GTCN')
parser.add_argument('--gpu', '-g', dest='gpu', default=1, type=int)
parser.add_argument('--seed', '-s', dest='seed', default=0, type=int)
parser.add_argument('--ckpt', dest='ckpt', required=True, help='path to pretrained .ckpt')
parser.add_argument('--yaml', dest='yaml', default='finetune.yaml', help='finetune yaml')
args = parser.parse_args()

gpu_ID = ",".join([str(i) for i in range(int(args.gpu))])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ID

lightning_lite.utilities.seed.seed_everything(seed=args.seed, workers=True)

yaml_path = osp.join(os.getcwd(), args.yaml)
config = utils.load_yaml(yaml_path)

# 1) build encoder skeleton, load pretrained weights
encoder = GTCN_ft(config)  
ckpt = torch.load(args.ckpt, map_location="cpu")
encoder.load_state_dict(ckpt["state_dict"], strict=True)

# 2) wrap into finetune model (freeze encoder, train head only)
model = GTCNFinetune(config=config, pretrained_encoder=encoder, freeze_encoder=True)

wandb_logger = WandbLogger(project='GTCN_finetune', name='ft_Eint', )
wandb_logger.watch(model, log='all', log_freq=50)


def train(config, data_path, Total_num, splitpara):
    train_dl, valid_dl = get_dl(config, data_path, Total_num, splitpara)
    print(f'train: {len(train_dl)} valid: {len(valid_dl)}')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_mae', mode='min')
    pcb = TQDMProgressBar(refresh_rate=4)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu,
        strategy='ddp' if args.gpu > 1 else 'auto',
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback, pcb],
        logger=wandb_logger,
        precision=16,
        gradient_clip_val=getattr(config, "gradient_clip_val", 0.0),
        gradient_clip_algorithm="value"
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    y_p = trainer.predict(model, valid_dl)
    y_p = torch.cat(y_p, dim=0)
    y = gety(valid_dl)
    score1 = F.l1_loss(y_p, y).item()
    print(f'Valid MAE: {score1:.4f}')
    print("weights saved:", trainer.log_dir)
    return {"VALID MAE": score1}


if __name__ == "__main__":
    dataset_path = '/home/xhwan/protein_data_processed'
    train(config=config, data_path=dataset_path, Total_num=300)
