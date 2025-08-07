# dataset/compute_latent_stats.py

import argparse, io, lmdb, pickle, yaml
from tqdm import tqdm
import numpy as np
import torch



def load_latent(raw_bytes: bytes, shape, dtype: np.dtype) -> np.ndarray:
    C, H, W = shape
    expect_nbytes = C * H * W * dtype.itemsize

    if len(raw_bytes) == expect_nbytes: # check if raw bytes 
        return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

    try:   # try torch.load
        obj = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    except Exception:
        obj = None

    if obj is None:
        try:   # try pickle.load
            obj = pickle.loads(raw_bytes)
        except Exception as e:
            raise RuntimeError(f"Cannot decode latent: {e}")

    # finally return a numpy array 
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        raise RuntimeError(f"Unsupported latent type: {type(obj)}")

    if arr.size != C * H * W:
        raise ValueError(f"Latent size {arr.shape} ≠ expected {(C, H, W)}")

    return arr.astype(dtype, copy=False).reshape(shape)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", required=True, help="Path to *.lmdb directory")
    parser.add_argument("--shape", type=int, nargs=3, default=[4, 16, 16],
                        metavar=("C", "H", "W"))
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "float64"])
    parser.add_argument("--output", help="Optional YAML to dump mean/std")
    args = parser.parse_args()

    C, H, W = args.shape
    dtype = np.dtype(args.dtype)

    sum_c, sumsq_c = np.zeros(C, np.float64), np.zeros(C, np.float64) # float64 precision 
    count = 0  # channel-wise counter 

    env = lmdb.open(args.lmdb, subdir=True, readonly=True,
                    lock=False, readahead=False)
    with env.begin(buffers=True) as txn:
        n_entries = txn.stat()["entries"]
        cursor = txn.cursor()
        for _, raw in tqdm(cursor, total=n_entries, desc="Scanning LMDB"):
            lat = load_latent(bytes(raw), (C, H, W), dtype).astype(np.float64)
            lat = lat.reshape(C, -1)          # (C, H*W)
            sum_c   += lat.sum(axis=1)
            sumsq_c += (lat ** 2).sum(axis=1)
            count   += H * W

    mean = sum_c / count
    std  = np.sqrt(sumsq_c / count - mean ** 2)

    print("Per-channel statistics (float64 precision)")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  channel[{i}]  mean = {m:.10f}   std = {s:.10f}")

    # save into a yaml file 
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump({"mean": mean.tolist(), "std": std.tolist()},
                      f, sort_keys=False)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

"""
Usage
-----
python compute_latent_stats.py \
    --lmdb  /scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents.lmdb \
    --shape 4 16 16 \
    --dtype float32 
"""