# main_train.py

import yaml
from dataset.font_dataset import FontPairDataset
from trainer.train_disentangle import train_loop
from trainer.train_DDPM import train_stage2_ddpm

def main():
    config = yaml.safe_load(open("configs/config.yaml"))
    dataset = FontPairDataset(root_dir=config['data_dir'], img_size=config['img_size'])

    if config.get("train_stage", 1) == 1:
        print("Training Stage 1: Joint training of encoder + decoder.")
        train_loop(config, dataset)
    elif config.get("train_stage") == 2:
        print("Training Stage 2: Freeze encoder, train DDPM decoder.")
        ckpt_path = config.get("freeze_ckpt", "")
        assert ckpt_path != "", "Missing `freeze_ckpt` path in config for Stage 2 training."
        train_stage2_ddpm(config, dataset, freeze_ckpt_path=ckpt_path)
    else:
        raise ValueError("`train_stage` must be either 1 or 2")

if __name__ == "__main__":
    main()