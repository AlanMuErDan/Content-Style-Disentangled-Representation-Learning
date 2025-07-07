# main_train.py

import yaml
from dataset.font_dataset import FontPairDataset
from trainer.train_disentangle import train_disentangle_loop
from trainer.train_vae import train_vae_loop

def main():
    config = yaml.safe_load(open("configs/config.yaml"))

    if config.get("train_stage") == "Disentangle":
        print("Disentangle VAE Training....")
        print(f"Training config: {config['disentangle']}")
        dataset = FontPairDataset(root_dir=config['data_dir'], img_size=config['img_size'])
        train_disentangle_loop(config["disentangle"], dataset)

    elif config.get("train_stage") == "VAE": 
        print("Disentangle VAE Training...")
        print(f"Training config: {config['vae']}")
        train_vae_loop(config["vae"])

    else:
        raise ValueError("`train_stage` must be either 1 or 2")

if __name__ == "__main__":
    main()