import argparse
import json
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "PingFang HK",
    "PingFang SC",
    "SimHei",
    "Heiti TC",
    "Songti SC",
    "STSong",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


DEFAULT_STYLE_PATH_FONTS = [
    "FZHeiJGTJW",
    "FZZH-NLHJW",
    "FZShangKJW",
    "FZYiKJW",
    "FZFWZhuZGDLMCJW",
]

DEFAULT_STYLE_PATH_CONTENTS = ["本", "木", "未"]


def parse_args():
    parser = argparse.ArgumentParser(description="Style-centric codebook analysis.")
    parser.add_argument(
        "--style-file",
        default="font_list.json",
        help="Path to JSON file listing style names (fonts).",
    )
    parser.add_argument(
        "--content-file",
        default="stroke_5_chars.txt",
        help="Path to text file listing content characters.",
    )
    parser.add_argument(
        "--codebook-path",
        default=os.path.join("base", "epoch_0500_S_codebook.pt"),
        help="Path to the torch tensor containing the codebook.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for saving figures/tables.",
    )
    parser.add_argument(
        "--style-path-fonts",
        nargs="+",
        default=DEFAULT_STYLE_PATH_FONTS,
        help="Ordered font names representing the style thickness path.",
    )
    parser.add_argument(
        "--style-path-contents",
        nargs="+",
        default=DEFAULT_STYLE_PATH_CONTENTS,
        help="Content characters used when visualizing the ordered style path.",
    )
    parser.add_argument(
        "--task3-num-styles",
        type=int,
        default=4,
        help="Number of styles sampled for the style-consistency analysis.",
    )
    parser.add_argument(
        "--task3-num-contents",
        type=int,
        default=4,
        help="Number of contents sampled for the style-consistency analysis.",
    )
    parser.add_argument(
        "--task3-seed",
        type=int,
        default=10086,
        help="Random seed for sampling styles/contents in Task 3.",
    )
    return parser.parse_args()


def load_lists(style_file, content_file):
    with open(style_file, "r") as f:
        style_list = json.load(f)

    with open(content_file, "r") as f:
        content_list = [line.rstrip() for line in f]

    return style_list, content_list


def load_codebook_tensor(codebook_path):
    tensor = torch.load(codebook_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor from {codebook_path}, got {type(tensor)}")
    return tensor.numpy()


def build_dataframe(style_list, content_list, coords_2d):
    num_styles = len(style_list)
    num_contents = len(content_list)
    expected = num_styles * num_contents
    if coords_2d.shape[0] != expected:
        raise ValueError(
            f"PCA-coordinates count {coords_2d.shape[0]} does not match style×content {expected}"
        )

    records = []
    for style_idx, style_name in enumerate(style_list):
        base = style_idx * num_contents
        for content_idx, content_char in enumerate(content_list):
            pt = coords_2d[base + content_idx]
            records.append(
                {
                    "style_idx": style_idx,
                    "style_name": style_name,
                    "content_idx": content_idx,
                    "content_char": content_char,
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                }
            )

    df = pd.DataFrame.from_records(records)
    df["flat_index"] = df["style_idx"] * num_contents + df["content_idx"]
    return df


def ensure_indices(style_list, requested_fonts):
    mapping = {name: idx for idx, name in enumerate(style_list)}
    indices = []
    missing = []
    for font in requested_fonts:
        if font in mapping:
            indices.append(mapping[font])
        else:
            missing.append(font)
    return indices, missing


def plot_style_paths(df, style_list, style_order, content_chars, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    style_indices, missing_fonts = ensure_indices(style_list, style_order)
    if missing_fonts:
        print(f"[Task 1] Warning: missing fonts {missing_fonts}. They will be skipped.")
    if len(style_indices) < 2:
        raise ValueError("At least two valid fonts are required for the style path visualization.")

    available_contents = [c for c in content_chars if c in df["content_char"].values]
    if not available_contents:
        raise ValueError("None of the requested contents are present in the dataset.")

    colors = matplotlib.colormaps.get_cmap("plasma")(
        np.linspace(0, 1, len(style_indices))
    )
    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="-", label=style_list[idx])
        for color, idx in zip(colors, style_indices)
    ]

    fig, axes = plt.subplots(
        1, len(available_contents), figsize=(6 * len(available_contents), 6)
    )
    if len(available_contents) == 1:
        axes = [axes]

    for ax, content_char in zip(axes, available_contents):
        rows = []
        for idx in style_indices:
            row = df[
                (df["style_idx"] == idx) & (df["content_char"] == content_char)
            ]
            if row.empty:
                rows = []
                print(
                    f"[Task 1] Missing combination: style {style_list[idx]} / content {content_char}. Skipping this content."
                )
                break
            rows.append(row.iloc[0])
        if not rows:
            ax.set_title(f"{content_char} (insufficient data)")
            continue
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        for i in range(len(rows) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="->", color=colors[i + 1], lw=2, alpha=0.85
                ),
            )
        for color, row in zip(colors, rows):
            ax.scatter(row["x"], row["y"], color=color, s=70)
        ax.set_title(f"{content_char} style trajectory")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    fig.suptitle("Ordered style trajectory per content (PCA 2D)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.subplots_adjust(bottom=0.25)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    path = os.path.join(output_dir, "style_paths.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Task 1] Saved style path visualization to {path}")


def run_style_consistency_analysis(
    df,
    style_list,
    content_list,
    original_embeddings,
    num_styles,
    num_contents,
    seed,
    output_dir,
):
    rng = np.random.default_rng(seed)
    unique_styles = df["style_idx"].unique()
    if len(unique_styles) < num_styles:
        num_styles = len(unique_styles)
    style_choices = rng.choice(unique_styles, size=num_styles, replace=False)

    available_contents = df["content_idx"].unique()
    if len(available_contents) < num_contents:
        num_contents = len(available_contents)
    content_choices = rng.choice(available_contents, size=num_contents, replace=False)

    subset = df[
        df["style_idx"].isin(style_choices) & df["content_idx"].isin(content_choices)
    ].copy()
    subset["style_name"] = subset["style_idx"].apply(lambda idx: style_list[idx])
    subset["content_char"] = subset["content_idx"].apply(lambda idx: content_list[idx])

    if subset.empty:
        raise ValueError("Subset for style consistency analysis is empty.")

    # Metrics computed in original embedding space (1024D)
    intra_dists = []
    style_groups = list(subset.groupby("style_idx"))
    for _, group in style_groups:
        indices = group["flat_index"].astype(int).values
        vectors = original_embeddings[indices]
        if vectors.shape[0] < 2:
            continue
        for i in range(vectors.shape[0]):
            for j in range(i + 1, vectors.shape[0]):
                intra_dists.append(float(np.linalg.norm(vectors[i] - vectors[j])))

    inter_dists = []
    for (_, group_a), (_, group_b) in combinations(style_groups, 2):
        idx_a = group_a["flat_index"].astype(int).values
        idx_b = group_b["flat_index"].astype(int).values
        vec_a = original_embeddings[idx_a]
        vec_b = original_embeddings[idx_b]
        for pa in vec_a:
            for pb in vec_b:
                inter_dists.append(float(np.linalg.norm(pa - pb)))

    intra_mean = float(np.mean(intra_dists)) if intra_dists else np.nan
    intra_std = float(np.std(intra_dists)) if intra_dists else np.nan
    inter_mean = float(np.mean(inter_dists)) if inter_dists else np.nan
    inter_std = float(np.std(inter_dists)) if inter_dists else np.nan
    separation_score = (
        inter_mean / intra_mean if intra_mean and inter_mean and intra_mean > 0 else np.nan
    )

    print("\n[Task 3] Style consistency metrics (computed in original 1024D space):")
    print(
        f"Intra-class mean={intra_mean:.4f} (std={intra_std:.4f}), "
        f"Inter-class mean={inter_mean:.4f} (std={inter_std:.4f}), "
        f"Separation score (inter/intra)={separation_score:.4f}"
    )

    # Visualization (PCA space)
    contents = sorted(subset["content_char"].unique())
    styles = sorted(subset["style_idx"].unique())
    cmap = matplotlib.colormaps.get_cmap("tab20")
    style_colors = [cmap(i / max(1, len(styles) - 1)) for i in range(len(styles))]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(9, 6))
    style_handles = {}
    content_handles = {}
    for _, row in subset.iterrows():
        style_pos = styles.index(row["style_idx"])
        content_pos = contents.index(row["content_char"])
        color = style_colors[style_pos]
        marker = markers[content_pos % len(markers)]
        ax.scatter(row["x"], row["y"], color=color, marker=marker, s=70, alpha=0.85)
        style_label = f"{row['style_name']} (#{row['style_idx']})"
        if style_label not in style_handles:
            style_handles[style_label] = plt.Line2D(
                [0], [0], color=color, marker="o", linestyle="", label=style_label
            )
        if row["content_char"] not in content_handles:
            content_handles[row["content_char"]] = plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle="",
                label=row["content_char"],
            )

    ax.set_title(
        f"Style clustering ({len(styles)} styles × {len(contents)} contents)"
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78, bottom=0.25)
    if style_handles:
        fig.legend(
            handles=list(style_handles.values()),
            title="Style (color)",
            loc="lower center",
            ncol=min(4, len(style_handles)),
            bbox_to_anchor=(0.5, 0.05),
            fontsize=8,
        )
    if content_handles:
        ax.legend(
            handles=list(content_handles.values()),
            title="Content (marker)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    path = os.path.join(output_dir, "style_consistency.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task 3] Saved style consistency visualization to {path}")

    metrics = {
        "num_styles": len(styles),
        "num_contents": len(contents),
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "separation_score": separation_score,
        "selected_style_indices": [int(idx) for idx in style_choices],
        "selected_style_names": [style_list[int(idx)] for idx in style_choices],
        "selected_content_indices": [int(idx) for idx in content_choices],
        "selected_content_chars": [content_list[int(idx)] for idx in content_choices],
        "embedding_space": "original_1024d",
    }
    metrics_path = os.path.join(output_dir, "style_consistency_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    style_list, content_list = load_lists(args.style_file, args.content_file)
    codebook = load_codebook_tensor(args.codebook_path)
    print(f"Loaded codebook with shape {codebook.shape}")

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(codebook)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance (2 components): {explained:.4f}")

    df = build_dataframe(style_list, content_list, coords_2d)

    plot_style_paths(
        df,
        style_list,
        args.style_path_fonts,
        args.style_path_contents,
        args.output_dir,
    )

    run_style_consistency_analysis(
        df,
        style_list,
        content_list,
        codebook,
        args.task3_num_styles,
        args.task3_num_contents,
        args.task3_seed,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
