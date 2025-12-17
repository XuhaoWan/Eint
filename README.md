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
