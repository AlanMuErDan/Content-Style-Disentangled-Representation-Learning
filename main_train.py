# main_train.py  

import yaml
from trainer.train_vae import train_vae_loop
from trainer.train_disentangle import train_disentangle_loop

cfg = yaml.safe_load(open("configs/config.yaml"))
stage = cfg.get("train_stage", "").strip().lower()

if stage == "vae":
    print("=== Stage-1: VAE Training ===")
    train_vae_loop(cfg["vae"])
elif stage == "disentangle":
    print("=== Stage-2: Disentangle + DDPM Training ===")
    train_disentangle_loop(cfg["disentangle"])
else:
    raise ValueError("config['train_stage'] must be 'VAE' or 'Disentangle'")

