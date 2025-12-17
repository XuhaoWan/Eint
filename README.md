# Physics-informed Pre-trained Graph Learning for Quantum Mechanics Interaction Energies Predictions of Ligand Screening

Core Developer: Xuhao Wan (xhwanrm@whu.edu.cn), Dr. Wei Yu (yuwei2@whu.edu.cn)

# GTCN: Pretraining on PubChemQC PM6 → Fine-tuning → Inference

This repo provides a simple workflow to:
1) download **PubChemQC PM6** (from the internet),
2) preprocess it into PyTorch Geometric graphs,
3) **pretrain** a GTCN encoder,
4) **fine-tune** GTCN (encoder frozen),
5) run **inference** with the fine-tuned checkpoint to predict QM interaction energy between protein and ligand.

## 0) Quick overview

- **Pretrain (large-scale)**: learn general physics representations from PubChemQC PM6
- **Fine-tune (task-specific)**: load pretrained weights, freeze the encoder, train a small readout head
- **Inference**: load fine-tuned checkpoint and predict on new data

## 1) Download PubChemQC PM6 

PubChemQC PM6 is very large (full set is **> 20TB**).
```bash
https://nakatamaho.riken.jp/pubchemqc.riken.jp/pm6_datasets.html
```

## 2) **Preprocess PubChemQC PM6 into this repo’s training format**
Then use the todata.py to preprocess the original dataset.

## 3) **Pretraining**
Create test.yaml (/pretrain.yaml) using the determined architecture.
Use your existing pretrain script (the one that builds model = GTCN(config) and calls trainer.fit(...)).
```bash
python train.py --gpu 8 --seed 0 --fold 0
```
After training, keep the best checkpoint produced by ModelCheckpoint(monitor="valid_mae").
You will use it as --ckpt for fine-tuning.

## 4) **Fine-tuning**
The next step is to obtain the protein–ligand pocket complex graph for fine-tuning and inference. Here, we implement this using the /utils/procomplex2graph.py file.
Then fine-tune YAML /train_ft.yaml (must match the pretrained encoder)
Important: these must match pretraining exactly, otherwise the checkpoint won’t load: max_fea_val, nd_fea, heads, hnrons, layers, E_node, E_edge, act, beta
Use the fine-tune training script:
```bash
python train_ft.py \
  --gpu 8 \
  --seed 0 \
  --ckpt /path/to/pretrained.ckpt \
  --yaml train_ft.yaml
```

## 5) **Inference (fine-tuned checkpoint)**
Run prediction using the fine-tuned checkpoint:
```bash
python finetune_predict.py \
  --gpu 8 \
  --seed 0 \
  --ckpt /path/to/finetuned.ckpt 
```
