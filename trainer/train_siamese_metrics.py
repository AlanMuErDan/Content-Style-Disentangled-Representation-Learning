
"""Full dataset training helpers."""
import argparse
import copy
import os
import random
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict, Any, List
import sys
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dataset.siamese_dataloading import (
    ImprovedLatentAccessor,
    FullDatasetPairDataset,
    latent_to_tensor,
)
from models.siamese_CNN import VGGEncoder, EnhancedContentEncoder, EnhancedStyleEncoder, SiameseJudge

DATA_PATHS = {
    "pt_path": "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents_v2_full.pt",
    "chars_path": "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/intersection_chars_full.txt",
    "fonts_json": "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_list_full.json",
    "vae_config": "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/configs/config.yaml",
    "vae_ckpt": "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/checkpoints/vae_best_ckpt.pth",
}

FULL_CONFIG = {
    "num_styles": 2056,
    "num_contents": 4574,
    "train_samples": 50000,
    "batch_size": 32,
    "epochs": 50,
    "lr": 5e-4,
    "eval_samples": 500,
    "val_samples": 512,
    "val_interval": 1,
    "pair_seed": 42,
    "pair_unique": True,
    "val_batch_size": 32,
    "checkpoint_dir": "siamese_checkpoint_enhanced_100",
}

from models import build_decoder


def _generate_pair_specs(
    task: Literal["content", "style"],
    num_contents: int,
    num_styles: int,
    total_pairs: int,
    seed: int,
    ensure_unique: bool = True,
    max_retry_factor: int = 20,
) -> List[Dict[str, Any]]:
    """Pre-generate pair specifications so train/val can share entities but not exact pairings."""
    if total_pairs <= 0:
        return []
    if num_contents < 2 or num_styles < 2:
        raise ValueError("Need at least two contents and styles to form pairs.")

    rng = random.Random(seed)
    specs: List[Dict[str, Any]] = []
    seen_pairs = set()

    max_attempts = max(total_pairs * max_retry_factor, 1000)
    attempts = 0

    while len(specs) < total_pairs and attempts < max_attempts:
        attempts += 1

        content_i = rng.randrange(num_contents)
        content_j = rng.randrange(num_contents)
        while content_j == content_i:
            content_j = rng.randrange(num_contents)

        style_p = rng.randrange(num_styles)
        style_q = rng.randrange(num_styles)
        while style_q == style_p:
            style_q = rng.randrange(num_styles)

        is_positive = rng.random() < 0.5

        if task == "content":
            pair_content = content_i if is_positive else content_j
            pair_style = style_q
        else:  # style task
            pair_content = content_j
            pair_style = style_p if is_positive else style_q

        key = (task, int(is_positive), content_i, style_p, pair_content, pair_style)
        if ensure_unique and key in seen_pairs:
            continue

        seen_pairs.add(key)
        specs.append(
            {
                "anchor_content_idx": content_i,
                "anchor_style_idx": style_p,
                "other_content_idx": content_j,
                "other_style_idx": style_q,
                "is_positive": is_positive,
            }
        )

    if len(specs) < total_pairs:
        raise RuntimeError(
            f"Unable to generate {total_pairs} unique pairs (got {len(specs)}). "
            "Consider reducing train/val sample counts or disabling uniqueness."
        )

    rng.shuffle(specs)
    return specs

def _load_config(path: str) -> dict:
    """Load YAML config"""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["vae"] if "vae" in cfg else cfg



def load_vae_decoder(
    config_path: str,
    ckpt_path: str,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict, torch.device]:
    """Load VAE decoder and config."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = _load_config(config_path)

    decoder = build_decoder(
        name=cfg["decoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    else:
        decoder.load_state_dict(ckpt)

    print(f"[INFO] Loaded VAE decoder on {device}")
    return decoder, cfg, device


def _tensor_to_wandb_image(wandb_module, tensor: torch.Tensor, caption: str):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return wandb_module.Image(array, caption=caption)

@torch.no_grad()
def score_pair(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor, device=None) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    logit = model(img1, img2)
    return torch.sigmoid(logit).item()


def build_sample_grid(
    accessor: ImprovedLatentAccessor,
    decoder: nn.Module,
    cfg: dict,
    device_used: torch.device,
    task: Literal["content", "style"],
    num_pairs: int = 4,
) -> torch.Tensor:
    """Create a grid of anchor/positive/negative samples for visualization."""

    def to_three_channel(img: torch.Tensor) -> torch.Tensor:
        if img.size(0) == 1:
            return img.repeat(3, 1, 1)
        return img

    tiles = []
    for _ in range(num_pairs):
        i = random.randrange(accessor.num_characters)
        j = random.randrange(accessor.num_characters)
        while j == i:
            j = random.randrange(accessor.num_characters)

        p = random.randrange(accessor.num_styles_per_char)
        q = random.randrange(accessor.num_styles_per_char)
        while q == p:
            q = random.randrange(accessor.num_styles_per_char)

        anchor = latent_to_tensor(decoder, accessor.get(i, p), cfg, device_used)

        if task == "content":
            positive = latent_to_tensor(decoder, accessor.get(i, q), cfg, device_used)
            negative = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used)
        else:
            positive = latent_to_tensor(decoder, accessor.get(j, p), cfg, device_used)
            negative = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used)

        tiles.extend(
            [
                to_three_channel(anchor.cpu()),
                to_three_channel(positive.cpu()),
                to_three_channel(negative.cpu()),
            ]
        )

    grid = make_grid(torch.stack(tiles), nrow=3, padding=2, normalize=True, value_range=(0, 1))
    return grid


def run_full_training(
    task: Literal["content", "style"],
    train_cfg: Dict[str, Any],
    encoder_type: Literal["enhanced", "vgg"] = "enhanced",
    device: Optional[str] = None,
    paths: Dict[str, Any] = DATA_PATHS,
    use_wandb: bool = False,
    wandb_project: str = "font-siamese",
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_run_name: Optional[str] = None,
    wandb_watch: bool = False,
    wandb_image_interval: int = 1,
    wandb_pairs_per_epoch: int = 12,
):
    """End-to-end training entry point."""
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)
    print(f"[train] task={task} encoder={encoder_type} device={device_obj}")
    print(f"[train] config={train_cfg}")

    wandb_module = None
    wandb_run = None
    if use_wandb:
        try:
            import wandb as wandb_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("wandb is not installed. Install it or set use_wandb=False.") from exc
        init_config = dict(train_cfg)
        init_config.update(
            {
                "task": task,
                "encoder_type": encoder_type,
                "device": device_str,
            }
        )
        if wandb_config:
            init_config.update(wandb_config)
        wandb_run = wandb_module.init(project=wandb_project, config=init_config, name=wandb_run_name)

    accessor = ImprovedLatentAccessor(
        pt_path=paths["pt_path"],
        chars_path=paths["chars_path"],
        fonts_json=paths["fonts_json"],
        device="cpu",
        latent_shape=(4, 16, 16),
    )

    decoder, cfg, decoder_device = load_vae_decoder(
        config_path=paths["vae_config"],
        ckpt_path=paths["vae_ckpt"],
        device=device_str,
    )

    total_contents = min(train_cfg["num_contents"], accessor.num_characters)
    total_styles = min(train_cfg["num_styles"], accessor.num_styles_per_char)
    train_samples = int(train_cfg["train_samples"])
    val_samples = max(0, int(train_cfg.get("val_samples", train_cfg.get("eval_samples", 0))))
    pair_seed = int(train_cfg.get("pair_seed", train_cfg.get("split_seed", 42)))
    ensure_unique_pairs = bool(train_cfg.get("pair_unique", True))

    total_pairs = train_samples + val_samples
    pair_specs = _generate_pair_specs(
        task=task,
        num_contents=total_contents,
        num_styles=total_styles,
        total_pairs=total_pairs,
        seed=pair_seed,
        ensure_unique=ensure_unique_pairs,
    )

    train_specs = pair_specs[:train_samples]
    if len(train_specs) < train_samples:
        raise RuntimeError(
            f"Requested {train_samples} training pairs but only generated {len(train_specs)}."
        )
    val_specs = pair_specs[train_samples:]
    if val_samples > 0 and len(val_specs) < val_samples:
        raise RuntimeError(
            f"Requested {val_samples} validation pairs but only generated {len(val_specs)}."
        )

    print(
        "[pairs] "
        f"train_pairs={len(train_specs)} val_pairs={len(val_specs)} | "
        f"num_contents={total_contents} num_styles={total_styles} | "
        f"unique_pairs={ensure_unique_pairs} seed={pair_seed}"
    )

    checkpoint_root = Path(train_cfg.get("checkpoint_dir", "siamese_checkpoint"))
    checkpoint_dir = checkpoint_root / task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] saving epoch checkpoints to {checkpoint_dir}")

    train_dataset = FullDatasetPairDataset(
        accessor=accessor,
        decoder=decoder,
        cfg=cfg,
        device_used=decoder_device,
        task=task,
        num_styles=total_styles,
        num_contents=total_contents,
        length=len(train_specs),
        pair_specs=train_specs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device_obj.type == "cuda",
    )

    val_loader: Optional[DataLoader] = None
    if val_specs:
        val_batch_size = int(train_cfg.get("val_batch_size", train_cfg["batch_size"]))
        val_dataset = FullDatasetPairDataset(
            accessor=accessor,
            decoder=decoder,
            cfg=cfg,
            device_used=decoder_device,
            task=task,
            num_styles=total_styles,
            num_contents=total_contents,
            length=len(val_specs),
            pair_specs=val_specs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device_obj.type == "cuda",
        )

    sample_img = latent_to_tensor(decoder, accessor.get(0, 0), cfg, decoder_device)

    model = SiameseJudge(
        in_ch=sample_img.shape[0],
        emb_dim=512,
        mlp_hidden=512,
        task=task,
        encoder_type=encoder_type,
    )

    if wandb_run and wandb_watch:
        wandb_module.watch(model, log="all", log_freq=100)

    def log_epoch_metrics(epoch: int, metrics: Dict[str, float], pair_samples) -> None:
        if wandb_run:
            log_payload = {"epoch": epoch}
            for key, value in metrics.items():
                log_payload[f"{task}/{key}"] = value
            wandb_module.log(log_payload, step=epoch)
            if pair_samples and wandb_pairs_per_epoch > 0:
                columns = [
                    "anchor_image",
                    "pair_image",
                    "label",
                    "anchor_content_idx",
                    "anchor_content_name",
                    "other_content_idx",
                    "other_content_name",
                    "anchor_style_idx",
                    "anchor_style_name",
                    "other_style_idx",
                    "other_style_name",
                    "is_positive",
                ]
                table = wandb_module.Table(columns=columns)
                for entry in pair_samples:
                    entry_data = dict(entry)
                    label_value = entry_data.get("label", 0.0)
                    anchor_img = _tensor_to_wandb_image(
                        wandb_module,
                        entry_data.pop("anchor_image"),
                        caption=f"anchor label={label_value:.1f}",
                    )
                    pair_img = _tensor_to_wandb_image(
                        wandb_module,
                        entry_data.pop("pair_image"),
                        caption=f"pair label={label_value:.1f}",
                    )
                    row = [
                        anchor_img,
                        pair_img,
                        label_value,
                        entry_data.get("anchor_content_idx"),
                        entry_data.get("anchor_content_name"),
                        entry_data.get("other_content_idx"),
                        entry_data.get("other_content_name"),
                        entry_data.get("anchor_style_idx"),
                        entry_data.get("anchor_style_name"),
                        entry_data.get("other_style_idx"),
                        entry_data.get("other_style_name"),
                        entry_data.get("is_positive"),
                    ]
                    table.add_data(*row)
                wandb_module.log({f"{task}/pair_table": table}, step=epoch)
            if wandb_image_interval > 0 and epoch % wandb_image_interval == 0:
                grid = build_sample_grid(accessor, decoder, cfg, decoder_device, task)
                wandb_module.log(
                    {f"{task}/samples": [wandb_module.Image(grid, caption=f"epoch {epoch} anchor/pos/neg")]},
                    step=epoch,
                )

    model, train_state = train_full_model(
        model,
        train_loader,
        device=device_str,
        lr=train_cfg["lr"],
        epochs=train_cfg["epochs"],
        epoch_logger=log_epoch_metrics if wandb_run else None,
        task_label=task,
        pair_log_limit=wandb_pairs_per_epoch if wandb_run else 0,
        val_loader=val_loader,
        val_interval=max(1, int(train_cfg.get("val_interval", 1))),
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=f"{task}",
    )

    best_val_metrics = train_state.get("best_val_metrics")
    if best_val_metrics:
        best_epoch = train_state.get("best_epoch")
        print(
            f"[train] best val @epoch {best_epoch}: "
            + " ".join(f"{k}={v:.4f}" for k, v in best_val_metrics.items())
        )
        best_ckpt_path = train_state.get("best_checkpoint_path")
        if best_ckpt_path:
            print(f"[train] best checkpoint path: {best_ckpt_path}")

    sanity_ok = quick_sanity_check(
        model, accessor, decoder, cfg, decoder_device, task, device_str, num_samples=50
    )

    metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if sanity_ok:
        metrics = eval_model(
            model,
            accessor,
            decoder,
            cfg,
            decoder_device,
            task,
            train_cfg["num_styles"],
            train_cfg["num_contents"],
            train_cfg["eval_samples"],
            device_str,
        )

    if wandb_run:
        if best_val_metrics:
            best_epoch = train_state.get("best_epoch")
            wandb_module.log(
                {f"{task}/{k}": v for k, v in best_val_metrics.items()},
                step=best_epoch if best_epoch is not None else train_cfg["epochs"],
            )
        wandb_module.log(
            {
                f"{task}/sanity_passed": float(sanity_ok),
                **{f"{task}/{k}": v for k, v in metrics.items()},
            },
            step=train_cfg["epochs"],
        )
        if wandb_image_interval <= 0 or train_cfg["epochs"] % wandb_image_interval != 0:
            grid = build_sample_grid(accessor, decoder, cfg, decoder_device, task)
            wandb_module.log(
                {f"{task}/samples": [wandb_module.Image(grid, caption=f"{task} anchor/pos/neg rows")]},
                step=train_cfg["epochs"],
            )
        wandb_run.finish()

    return model, accessor, decoder, metrics


def evaluate_on_loader(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: str = "cuda",
    criterion: Optional[nn.Module] = None,
    metric_prefix: str = "val",
) -> Dict[str, float]:
    """Evaluate siamese model on a dataloader and return aggregated metrics."""
    if loader is None:
        return {}

    model.eval()
    device_obj = torch.device(device)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x1, x2, y = batch[:3]
            else:
                raise ValueError("Expected batch to provide (x1, x2, y).")

            x1 = x1.to(device_obj)
            x2 = x2.to(device_obj)
            y = y.to(device_obj)

            logits = model(x1, x2)
            if criterion is not None:
                total_loss += criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch_acc = (preds == y).float().mean().item()
            total_acc += batch_acc

            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches and criterion is not None else 0.0
    avg_acc = total_acc / num_batches if num_batches else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        f"{metric_prefix}_loss": avg_loss,
        f"{metric_prefix}_acc": avg_acc,
        f"{metric_prefix}_precision": precision,
        f"{metric_prefix}_recall": recall,
        f"{metric_prefix}_f1": f1,
    }
    return metrics


def train_full_model(
    model,
    loader,
    device="cuda",
    lr=1e-4,
    epochs=20,
    epoch_logger=None,
    task_label: str = "train",
    pair_log_limit: int = 0,
    val_loader: Optional[DataLoader] = None,
    val_interval: int = 1,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: Optional[str] = None,
):
    """Basic supervised training loop with optional validation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    val_interval = max(1, int(val_interval)) if val_loader is not None else 1
    best_state_dict = None
    best_val_metrics: Optional[Dict[str, float]] = None
    best_val_score = float("-inf")
    best_epoch: Optional[int] = None
    last_val_metrics: Optional[Dict[str, float]] = None
    checkpoint_dir_path: Optional[Path] = None
    if checkpoint_dir is not None:
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = (checkpoint_prefix or task_label).replace(" ", "_")
    saved_checkpoints: List[str] = []
    best_checkpoint_path: Optional[str] = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        tp = fp = tn = fn = 0
        epoch_pair_samples = []

        batch_iter = tqdm(
            loader,
            desc=f"[train-{task_label}] epoch {ep}/{epochs}",
            leave=False,
        )
        for batch in batch_iter:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                x1, x2, y, meta = batch
            else:
                x1, x2, y = batch
                meta = None

            if pair_log_limit > 0 and meta is not None and len(epoch_pair_samples) < pair_log_limit:
                bsz = y.size(0)
                take = min(pair_log_limit - len(epoch_pair_samples), bsz)
                for idx in range(take):
                    sample_meta = {}
                    for key, value in meta.items():
                        if isinstance(value, torch.Tensor):
                            sample_meta[key] = value[idx].item()
                        elif isinstance(value, (list, tuple)):
                            sample_meta[key] = value[idx]
                        else:
                            sample_meta[key] = value
                    sample_meta["label"] = float(y[idx].item())
                    sample_meta["anchor_image"] = x1[idx].detach().cpu()
                    sample_meta["pair_image"] = x2[idx].detach().cpu()
                    epoch_pair_samples.append(sample_meta)
                    if len(epoch_pair_samples) >= pair_log_limit:
                        break

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                batch_acc = (preds == y).float().mean().item()
                total_acc += batch_acc
                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

            total_loss += loss.item()
            num_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

        avg_loss = total_loss / num_batches if num_batches else 0.0
        avg_acc = total_acc / num_batches if num_batches else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        train_metrics = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
        }

        val_metrics: Dict[str, float] = {}
        should_validate = val_loader is not None and (ep % val_interval == 0 or ep == epochs)
        is_best = False
        if should_validate:
            val_metrics = evaluate_on_loader(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                metric_prefix="val",
            )
            last_val_metrics = val_metrics
            current_score = val_metrics.get("val_f1", 0.0)
            if current_score > best_val_score:
                best_val_score = current_score
                best_state_dict = copy.deepcopy(model.state_dict())
                best_val_metrics = dict(val_metrics)
                best_epoch = ep
                is_best = True

        log_msg = (
            f"[train] epoch {ep}/{epochs} loss={avg_loss:.4f} acc={avg_acc:.4f} "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )
        if val_metrics:
            log_msg += (
                f" | val_loss={val_metrics.get('val_loss', 0.0):.4f}"
                f" val_acc={val_metrics.get('val_acc', 0.0):.4f}"
                f" val_f1={val_metrics.get('val_f1', 0.0):.4f}"
            )
        print(log_msg)

        combined_metrics = dict(train_metrics)
        combined_metrics.update(val_metrics)
        if epoch_logger:
            epoch_logger(ep, combined_metrics, epoch_pair_samples)

        if checkpoint_dir_path is not None:
            ckpt_path = checkpoint_dir_path / f"{checkpoint_prefix}_epoch_{ep:04d}.pth"
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics if val_metrics else None,
                    "config": {
                        "device": device,
                        "lr": lr,
                        "task_label": task_label,
                    },
                },
                ckpt_path,
            )
            saved_checkpoints.append(str(ckpt_path))
            if is_best:
                best_checkpoint_path = str(ckpt_path)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, {
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "last_val_metrics": last_val_metrics,
        "checkpoint_paths": saved_checkpoints,
        "best_checkpoint_path": best_checkpoint_path,
    }

def eval_model(model, accessor, decoder, cfg, device_used, task, num_styles, num_contents, eval_samples, device):
    """Evaluate model accuracy on randomly decoded pairs."""
    model.eval()
    device_obj = torch.device(device)

    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    test_contents = list(range(num_contents // 2, num_contents))
    test_styles = list(range(num_styles // 2, num_styles))

    with torch.no_grad():
        for _ in tqdm(
            range(eval_samples),
            desc=f"[eval-{task}]",
            leave=False,
        ):
            i = random.choice(test_contents)
            j = random.choice(test_contents)
            while j == i:
                j = random.choice(test_contents)

            p = random.choice(test_styles)
            q = random.choice(test_styles)
            while q == p:
                q = random.choice(test_styles)

            img_i_p = latent_to_tensor(decoder, accessor.get(i, p), cfg, device_used).to(device_obj)

            if task == "content":
                img_pos = latent_to_tensor(decoder, accessor.get(i, q), cfg, device_used).to(device_obj)
                img_neg = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used).to(device_obj)
            else:
                img_pos = latent_to_tensor(decoder, accessor.get(j, p), cfg, device_used).to(device_obj)
                img_neg = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used).to(device_obj)

            pos_logit = model(img_i_p.unsqueeze(0), img_pos.unsqueeze(0))
            neg_logit = model(img_i_p.unsqueeze(0), img_neg.unsqueeze(0))

            pos_prob = torch.sigmoid(pos_logit).item()
            neg_prob = torch.sigmoid(neg_logit).item()

            if pos_prob > 0.5:
                tp += 1
            else:
                fn += 1

            if neg_prob > 0.5:
                fp += 1
            else:
                tn += 1

            if pos_prob > neg_prob:
                correct += 1
            total += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(
        f"[eval] task={task} samples={total} "
        f"acc={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def quick_sanity_check(model, accessor, decoder, cfg, device_used, task, device, num_samples=50):
    """Quick sanity test to confirm the model learned basic distinctions."""
    model.eval()
    device_obj = torch.device(device)

    same_scores = []
    diff_scores = []
    correct = 0

    with torch.no_grad():
        for _ in tqdm(
            range(num_samples),
            desc=f"[sanity-{task}]",
            leave=False,
        ):
            i = random.randrange(accessor.num_characters)
            j = random.randrange(accessor.num_characters)
            while j == i:
                j = random.randrange(accessor.num_characters)

            p = random.randrange(accessor.num_styles_per_char)
            q = random.randrange(accessor.num_styles_per_char)
            while q == p:
                q = random.randrange(accessor.num_styles_per_char)

            img_anchor = latent_to_tensor(decoder, accessor.get(i, p), cfg, device_used).to(device_obj)

            if task == "content":
                img_same = latent_to_tensor(decoder, accessor.get(i, q), cfg, device_used).to(device_obj)
                img_diff = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used).to(device_obj)
            else:
                img_same = latent_to_tensor(decoder, accessor.get(j, p), cfg, device_used).to(device_obj)
                img_diff = latent_to_tensor(decoder, accessor.get(j, q), cfg, device_used).to(device_obj)

            same_prob = torch.sigmoid(model(img_anchor.unsqueeze(0), img_same.unsqueeze(0))).item()
            diff_prob = torch.sigmoid(model(img_anchor.unsqueeze(0), img_diff.unsqueeze(0))).item()

            same_scores.append(same_prob)
            diff_scores.append(diff_prob)
            if same_prob > diff_prob:
                correct += 1

    avg_same = sum(same_scores) / len(same_scores) if same_scores else 0.0
    avg_diff = sum(diff_scores) / len(diff_scores) if diff_scores else 0.0
    sanity_acc = correct / num_samples if num_samples else 0.0

    print(f"[sanity] task={task} same={avg_same:.3f} diff={avg_diff:.3f} acc={sanity_acc:.3f}")
    return sanity_acc > 0.7


def full_scale_training(encoder_type="enhanced", **run_kwargs):
    """Full dataset training helper."""
    print(f"[train] full-scale run encoder={encoder_type}")
    content_model, accessor, decoder, content_metrics = run_full_training(
        task="content",
        train_cfg=FULL_CONFIG,
        encoder_type="enhanced",
        **run_kwargs,
    )
    torch.save(content_model.state_dict(), "content_siamese_model_enhanced.pth")
    print(f"[train] content metrics: {content_metrics}")

    style_model, _, _, style_metrics = run_full_training(
        task="style",
        train_cfg=FULL_CONFIG,
        encoder_type="enhanced",
        **run_kwargs,
    )
    torch.save(style_model.state_dict(), "style_siamese_model_full_enhanced.pth")
    print(f"[train] style metrics: {style_metrics}")

    return content_model, style_model, accessor, decoder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train siamese metrics model")
    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="full",
        help="debug=quick sanity run, full=full-scale training",
    )
    parser.add_argument(
        "--encoder",
        choices=["enhanced", "vgg"],
        default="enhanced",
        help="Encoder backbone to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="font-siamese",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name",
    )
    parser.add_argument(
        "--wandb-watch",
        action="store_true",
        help="Watch model gradients/parameters in W&B",
    )
    parser.add_argument(
        "--wandb-image-interval",
        type=int,
        default=1,
        help="Log sample grid to W&B every N epochs (default 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--wandb-pairs-per-epoch",
        type=int,
        default=12,
        help="Number of pair samples to log to W&B per epoch (default 12, 0 disables).",
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=None,
        help="Append one or more tags to W&B config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb_config = {
        "mode": args.mode,
        "encoder": args.encoder,
    }
    if args.wandb_tag:
        wandb_config["tags"] = args.wandb_tag

    run_kwargs = {
        "device": args.device,
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_watch": args.wandb_watch,
        "wandb_config": wandb_config if args.wandb else None,
        "wandb_image_interval": max(args.wandb_image_interval, 0),
        "wandb_pairs_per_epoch": max(args.wandb_pairs_per_epoch, 0),
    }

    print("[cli] running full-scale training")
    full_scale_training(encoder_type=args.encoder, **run_kwargs)


if __name__ == "__main__":
    main()
