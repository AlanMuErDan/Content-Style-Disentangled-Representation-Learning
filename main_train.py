# main_train.py  （放在项目根目录）
# -------------------------------------------------
# 统一入口：根据 config["train_stage"] 选择训练脚本
#   - "VAE"          → trainer.train_vae.train_vae_loop(cfg["vae"])
#   - "Disentangle"  → trainer.train_disentangle.train(cfg["disentangle"])
# -------------------------------------------------
import yaml

# stage-1  (VAE)
from trainer.train_vae import train_vae_loop

# stage-2  (Disentangle + DDPM)
import trainer.train_disentangle as _td   # 直接 import 模块
# 兼容：如果脚本里后来改名成 train_disentangle_loop 也能用
_train_disentangle = (
    getattr(_td, "train_disentangle_loop", None) or getattr(_td, "train")
)

def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    stage = cfg.get("train_stage", "").strip()

    if stage.lower() == "vae":
        print("=== Stage-1: VAE Training ===")
        train_vae_loop(cfg["vae"])

    elif stage.lower() == "disentangle":
        print("=== Stage-2: Disentangle + DDPM Training ===")
        _train_disentangle(cfg["disentangle"])

    else:
        raise ValueError("config['train_stage'] 必须设为 'VAE' 或 'Disentangle'")

if __name__ == "__main__":
    main()