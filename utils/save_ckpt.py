# utils/save_ckpt.py

import os
import time
import json
import torch
from typing import Optional, Dict, Any

__all__ = [
    "create_experiment_dir",
    "build_state_dict",
    "CheckpointManager"
]


def create_experiment_dir(config: Dict[str, Any],
                          base_dir: str = "checkpoints",
                          exp_name_key: str = "experiment_name") -> str:
    os.makedirs(base_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = config.get(exp_name_key) or config.get("train_stage") or "exp"
    seed = config.get("seed", "NA")
    dir_name = f"{ts}_{name}_seed{seed}"
    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config_snapshot.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CheckpointManager] 写 config_snapshot.json 失败: {e}")
    print(f"[CheckpointManager] Experiment directory: {exp_dir}")
    return exp_dir


def _to_cpu_state_dict(module):
    if module is None:
        return None
    try:
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}
    except Exception:
        if hasattr(module, "named_parameters"):
            return {k: p.detach().cpu() for k, p in module.named_parameters()}
        raise


def build_state_dict(encoder,
                     decoder,
                     vq,
                     gen_optim,
                     gen_scheduler,
                     epoch: int,
                     global_step: int,
                     ema_encoder=None,
                     ema_decoder=None,
                     ema_vq=None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder": _to_cpu_state_dict(encoder),
        "decoder": _to_cpu_state_dict(decoder),
        "vq": _to_cpu_state_dict(vq),
        "gen_optim": gen_optim.state_dict() if gen_optim is not None else None,
        "gen_scheduler": gen_scheduler.state_dict() if gen_scheduler is not None else None,
        "ema": {
            "encoder": _to_cpu_state_dict(ema_encoder),
            "decoder": _to_cpu_state_dict(ema_decoder),
            "vq": _to_cpu_state_dict(ema_vq)
        } if (ema_encoder or ema_decoder or ema_vq) else None
    }
    if extra:
        state.update(extra)
    return state


class CheckpointManager:
    def __init__(self,
                 exp_dir: str,
                 monitor: str = "val_recon_l2",
                 mode: str = "min",
                 best_filename: str = "best_ckpt.pth",
                 last_filename: str = "last_ckpt_epoch.pth",
                 rank: int = 0):
        self.exp_dir = exp_dir
        self.monitor = monitor
        assert mode in ("min", "max")
        self.mode = mode
        self.best_filename = best_filename
        self.last_filename = last_filename
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.rank = rank

        self._meta_path = os.path.join(self.exp_dir, "ckpt_meta.json")
        self._try_restore_meta()

    def save_last(self, state: Dict[str, Any]):
        if self.rank != 0:
            return
        path = os.path.join(self.exp_dir, self.last_filename)
        self._atomic_save(state, path)
        self._write_meta(update={
            "last_path": path,
            "last_epoch": state.get("epoch"),
            "last_global_step": state.get("global_step")
        })
        print(f"[CheckpointManager] Updated last checkpoint: {path}")

    def update_best(self, metric_value: float, state: Dict[str, Any]):
        improved = self._is_improved(metric_value)
        self.best_value = metric_value
        self.best_epoch = state.get("epoch", -1)
        if self.rank == 0:
            path = os.path.join(self.exp_dir, self.best_filename)
            state_with_metric = dict(state)
            state_with_metric["best_metric"] = {
                "name": self.monitor,
                "value": metric_value,
                "epoch": self.best_epoch 
            }
            self._atomic_save(state_with_metric, path)
            self._write_meta(update={
                "best_path": path,
                "best_value": metric_value,
                "best_epoch": self.best_epoch
            })
            status = "IMPROVED" if improved else "UPDATED"
            print(f"[CheckpointManager] {status} best checkpoint: {path} "
                  f"({self.monitor}={metric_value:.6f}, epoch={self.best_epoch})")

    def maybe_update_best(self, metric_value: float, state: Dict[str, Any]):
        if self._is_improved(metric_value):
            self.update_best(metric_value, state)

    def _is_improved(self, metric_value: float) -> bool:
        if self.mode == "min":
            return metric_value < self.best_value
        return metric_value > self.best_value

    def _atomic_save(self, obj: Dict[str, Any], path: str):
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path) 

    def _try_restore_meta(self):
        if os.path.isfile(self._meta_path):
            try:
                import json
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "best_value" in meta and "best_epoch" in meta:
                    self.best_value = meta["best_value"]
                    self.best_epoch = meta["best_epoch"]
                print(f"[CheckpointManager] Restored meta: best={self.best_value} "
                      f"epoch={self.best_epoch}")
            except Exception as e:
                print(f"[CheckpointManager] Failed to read meta file: {e}")

    def _write_meta(self, update: Dict[str, Any]):
        if self.rank != 0:
            return
        meta = {}
        if os.path.isfile(self._meta_path):
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        meta.update(update)
        meta.setdefault("best_value", self.best_value)
        meta.setdefault("best_epoch", self.best_epoch)
        try:
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CheckpointManager] Failed to write meta file: {e}")
