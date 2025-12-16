import os, torch
from VAE_inference import merge_pt_shards_to_single_pt

shard_dir   = SHARD_PATH
out_pt_path = OUT_PT_PATH

local_tmp = os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER','tmp')}"
merge_pt_shards_to_single_pt(shard_dir, out_pt_path, dtype="float16",
                             tmp_dir=local_tmp, flush_every_gb=4)

pkg = torch.load(out_pt_path, map_location="cpu")
lat = pkg["latents"]
print("merged:", lat.shape, lat.dtype)