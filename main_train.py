import yaml
from dataset.font_dataset import FontPairDataset
from trainer.train import train_loop


def main():
    config = yaml.safe_load(open("configs/config.yaml"))
    dataset = FontPairDataset(root_dir=config['data_dir'], img_size=config['img_size'])
    train_loop(config, dataset)

if __name__ == "__main__":
    main()