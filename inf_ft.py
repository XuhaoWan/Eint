import argparse
import os
import os.path as osp
import lightning_lite
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils import utils
from data.data import get_dl, gety
from models.GTCN import GTCN_ft
from models.GTCN_ft import GTCNFinetune


def main():
    parser = argparse.ArgumentParser(description='Predict with fine-tuned GTCN')
    parser.add_argument('--gpu', '-g', dest='gpu', default=1, type=int)
    parser.add_argument('--seed', '-s', dest='seed', default=0, type=int)

    parser.add_argument('--ckpt', dest='ckpt', required=True, help='path to fine-tuned .ckpt')

    parser.add_argument('--yaml', dest='yaml', default='finetune.yaml', help='yaml used in finetune')
    parser.add_argument('--data_path', dest='data_path', required=True, help='dataset path')
    parser.add_argument('--total_num', dest='total_num', default=10, type=int)

    parser.add_argument('--split_train', dest='split_train', default=0, type=int)
    parser.add_argument('--split_pred', dest='split_pred', default=10, type=int)

    parser.add_argument('--out', dest='out', default='predictions.pt', help='output file (.pt)')
    args = parser.parse_args()

    # GPU visible
    gpu_ID = ",".join([str(i) for i in range(int(args.gpu))])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ID

    lightning_lite.utilities.seed.seed_everything(seed=args.seed, workers=True)

    yaml_path = osp.join(os.getcwd(), args.yaml)
    config = utils.load_yaml(yaml_path)

    # 1) build encoder + finetune wrapper 
    encoder = GTCN_ft(config)
    model = GTCNFinetune(config=config, pretrained_encoder=encoder, freeze_encoder=True)

    # 2) load fine-tuned weights into the WHOLE finetune model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # 3) build pred dataloader
    splitpara = [args.split_train, args.split_pred]
    _, pred_dl = get_dl(config, args.data_path, args.total_num, splitpara)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu,
        strategy='ddp' if args.gpu > 1 else 'auto',
        logger=False,
        enable_checkpointing=False,
        precision=16,
    )

    # 4) predict
    preds = trainer.predict(model, dataloaders=pred_dl)
    preds = torch.cat(preds, dim=0).detach().cpu()

    # 5) 
    mae = None
    try:
        y_true = gety(pred_dl).detach().cpu()
        mae = F.l1_loss(preds, y_true).item()
        print(f"Predict MAE: {mae:.6f}")
    except Exception as e:
        print(f"Skip MAE (no labels or gety failed): {e}")

    # 6) save
    payload = {"pred": preds}
    if mae is not None:
        payload["mae"] = mae
    torch.save(payload, args.out)
    print(f"Saved predictions to: {args.out}")


if __name__ == "__main__":
    main()
