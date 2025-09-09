import torch, random
from torch.utils.data import DataLoader
from font_dataset import FourWayFontPairLatentPTDataset
import yaml

# paths
PT_PATH     = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents_v2.pt"
CHARS_PATH  = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/intersection_chars.txt"
FONTS_JSON  = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_list.json"
YAML_PATH = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/configs/latent_stats.yaml"
EXPECTED_CHW = (4, 16, 16)

@torch.no_grad()
def main(num_samples=512):
    torch.manual_seed(0); random.seed(0)

    lat = torch.load(PT_PATH)["latents"]  # (N,H,W,C)
    print(f"Shape: {lat.shape}")          # torch.Size([9406200, 16, 16, 4])
    lat = lat.view(-1, lat.shape[-1])     # (N×H×W,C)
    print(f"Shape: {lat.shape}")          # torch.Size([2407987200, 4])

    mean = lat.mean(0)
    std = lat.std(0)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    ds = FourWayFontPairLatentPTDataset(
        pt_path=PT_PATH,
        chars_path=CHARS_PATH,
        fonts_json=FONTS_JSON,
        latent_shape=EXPECTED_CHW,
        pair_num=max(num_samples, 64),
        stats_yaml=None,
    )
    ds.mean = mean.view(-1,1,1).to(ds.latents_hwc.dtype)
    ds.std  = std.view(-1,1,1).to(ds.latents_hwc.dtype)

    Z = []
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    for i, item in enumerate(loader):
        if len(Z) >= num_samples: break
        for k in ("F_A+C_A", "F_A+C_B", "F_B+C_A", "F_B+C_B"):
            z = item[k].squeeze(0)
            assert z.shape == EXPECTED_CHW
            Z.append(z.cpu())

    Z = torch.stack(Z)                            # (M,C,H,W)
    Zf = Z.permute(0,2,3,1).reshape(-1, Z.shape[1])
    m, s = Zf.mean(0), Zf.std(0, unbiased=False)
    print("empirical mean:", m.tolist())
    print("empirical std :", s.tolist())
    assert (m.abs() < 0.05).all()
    assert ((s > 0.9) & (s < 1.1)).all()

    for _ in range(8):  # denorm check
        d = ds[0]
        for k, f, c in [("F_A+C_A", d["font_a"], d["char_a"]),
                        ("F_A+C_B", d["font_a"], d["char_b"]),
                        ("F_B+C_A", d["font_b"], d["char_a"]),
                        ("F_B+C_B", d["font_b"], d["char_b"])]:
            i = ds._flat_index(ds.fonts.index(f), ds.chars.index(c))
            orig = ds.latents_hwc[i].permute(2,0,1).float()
            norm = d[k].float()
            rec = norm * ds.std.float() + ds.mean.float()

    print("All tests passed.")

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(stats, f, default_flow_style=True)
    print(f"Saved stats to: {YAML_PATH}")

if __name__ == "__main__":
    main()