# main_train.py  

import yaml
from trainer.train_vae import train_vae_loop
from trainer.train_disentangle_mar import train_disentangle_loop_mar
from trainer.train_disentangle_sd import train_disentangle_loop_sd
from trainer.train_disentangle_sd_pro import train_disentangle_loop_sd_pro
from trainer.train_disentangle_regression import train_disentangle_regression_loop

cfg = yaml.safe_load(open("configs/config.yaml"))
stage = cfg.get("train_stage", "").strip().lower()

if stage == "vae":
    print("=== Stage-1: VAE Training ===")
    train_vae_loop(cfg["vae"])
elif stage == "disentangle_mar":
    print("=== Stage-2: Disentangle + DDPM MAR Training ===")
    train_disentangle_loop_mar(cfg["disentangle_mar"])
elif stage == "disentangle_regression":
    print("=== Stage-2: Disentangle + Regression Training ===")
    train_disentangle_regression_loop(cfg["disentangle_regression"])
elif stage == "disentangle_sd":
    print("=== Stage-2: Disentangle + DDPM SD Training ===")
    train_disentangle_loop_sd(cfg["disentangle_sd"])
elif stage == "disentangle_sd_pro":
    print("=== Stage-2: Disentangle + DDPM SD PRO Training ===")
    train_disentangle_loop_sd_pro(cfg["disentangle_sd_pro"])
else:
    raise ValueError("config['train_stage'] must be 'VAE' or 'Disentangle'")

