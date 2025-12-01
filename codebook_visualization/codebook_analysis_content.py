import argparse
import json
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")  # Ensure headless-friendly rendering
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


SEMANTIC_SCENARIOS = [
    {
        "name": "犬_vs_尤_stroke",
        "description": "Stroke analogy: 大→犬 vs 尢→尤",
        "pairs": [("大", "犬"), ("尢", "尤")],
    },
    {
        "name": "天_vs_正_stroke",
        "description": "Stroke analogy: 大→天 vs 止→正",
        "pairs": [("大", "天"), ("止", "正")],
    },
]

def parse_args():
    parser = argparse.ArgumentParser(description="Codebook visualization utilities.")
    parser.add_argument(
        "--style-file",
        default="font_list.json",
        help="Path to JSON file that lists style names (fonts).",
    )
    parser.add_argument(
        "--content-file",
        default="stroke_5_chars.txt",
        help="Path to text file that lists content characters.",
    )
    parser.add_argument(
        "--codebook-path",
        default=os.path.join("base", "epoch_0500_C_codebook.pt"),
        help="Path to the torch tensor containing the codebook.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory that will store figures/tables.",
    )
    parser.add_argument(
        "--vector-plot-styles",
        type=int,
        nargs="*",
        default=list(range(4)),
        help="Style indices used for the semantic vector visualization.",
    )
    parser.add_argument(
        "--task3-num-styles",
        type=int,
        default=4,
        help="Number of styles sampled for the content consistency analysis.",
    )
    parser.add_argument(
        "--task3-num-contents",
        type=int,
        default=4,
        help="Number of contents sampled for the content consistency analysis.",
    )
    parser.add_argument(
        "--task3-seed",
        type=int,
        default=23,
        help="Random seed for style/content sampling in task 3.",
    )
    return parser.parse_args()


def load_lists(style_file, content_file):
    with open(style_file, "r") as f:
        style_list = json.load(f)

    with open(content_file, "r") as f:
        contents = [line.rstrip() for line in f]

    return style_list, contents


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
            f"Flattened PCA array has {coords_2d.shape[0]} rows but expected {expected}"
        )

    records = []
    for style_idx, style_name in enumerate(style_list):
        base = style_idx * num_contents
        for content_idx, content_char in enumerate(content_list):
            point = coords_2d[base + content_idx]
            records.append(
                {
                    "style_idx": style_idx,
                    "style_name": style_name,
                    "content_idx": content_idx,
                    "content_char": content_char,
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
            )

    df = pd.DataFrame.from_records(records)
    lookup = {
        (int(row.style_idx), row.content_char): np.array([row.x, row.y], dtype=float)
        for row in df.itertuples()
    }
    return df, lookup


def cosine_similarity(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return np.nan
    return float(np.dot(vec_a, vec_b) / denom)


def run_semantic_shift_analysis(df, lookup, style_list, style_indices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    style_indices = [idx for idx in style_indices if 0 <= idx < len(style_list)]
    if not style_indices:
        style_indices = list(range(min(6, len(style_list))))

    scenario_stats = []
    for scenario in SEMANTIC_SCENARIOS:
        cos_values = []
        for style_idx in df["style_idx"].unique():
            vectors = []
            for src, dst in scenario["pairs"]:
                key_src = (int(style_idx), src)
                key_dst = (int(style_idx), dst)
                if key_src not in lookup or key_dst not in lookup:
                    vectors = []
                    break
                vectors.append(lookup[key_dst] - lookup[key_src])
            if len(vectors) == len(scenario["pairs"]):
                cos = cosine_similarity(vectors[0], vectors[1])
                if not np.isnan(cos):
                    cos_values.append(cos)

        scenario_stats.append(
            {
                "scenario": scenario["description"],
                "styles_used": len(cos_values),
                "mean_cosine": float(np.mean(cos_values)) if cos_values else np.nan,
                "std_cosine": float(np.std(cos_values)) if cos_values else np.nan,
            }
        )

    stats_df = pd.DataFrame(scenario_stats)
    print("\n[Task 1] Stroke semantics cosine similarities:")
    print(stats_df.to_string(index=False))

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    fig, axes = plt.subplots(1, len(SEMANTIC_SCENARIOS), figsize=(7 * len(SEMANTIC_SCENARIOS), 6))
    if len(SEMANTIC_SCENARIOS) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, SEMANTIC_SCENARIOS):
        pair_cmap = matplotlib.colormaps.get_cmap("Set1")
        pair_colors = [
            pair_cmap(i / max(1, len(scenario["pairs"]) - 1)) for i in range(len(scenario["pairs"]))
        ]
        style_handles = {}
        pair_handles = {}
        for style_pos, style_idx in enumerate(style_indices):
            style_name = style_list[style_idx]
            missing = False
            for src, dst in scenario["pairs"]:
                if (style_idx, src) not in lookup or (style_idx, dst) not in lookup:
                    missing = True
                    break
            if missing:
                continue
            marker = marker_cycle[style_pos % len(marker_cycle)]
            style_label = f"{style_name} (#{style_idx})"
            if style_label not in style_handles:
                style_handles[style_label] = plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="",
                    color="black",
                    label=style_label,
                    markersize=8,
                )
            for pair_idx, (src, dst) in enumerate(scenario["pairs"]):
                src_coord = lookup[(style_idx, src)]
                dst_coord = lookup[(style_idx, dst)]
                color = pair_colors[pair_idx]
                pair_label = f"{src}→{dst}"
                if pair_label not in pair_handles:
                    pair_handles[pair_label] = plt.Line2D(
                        [0], [0], color=color, lw=2, label=pair_label
                    )
                ax.scatter(
                    [src_coord[0]],
                    [src_coord[1]],
                    color=color,
                    marker=marker,
                    s=50,
                    alpha=0.9,
                )
                ax.scatter(
                    [dst_coord[0]],
                    [dst_coord[1]],
                    color=color,
                    marker=marker,
                    s=70,
                    alpha=0.9,
                )
                ax.annotate(
                    "",
                    xy=dst_coord,
                    xytext=src_coord,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.8),
                )
        ax.set_title(f"{scenario['description']}")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        if pair_handles:
            pair_legend = ax.legend(
                handles=list(pair_handles.values()), title="Transform (color)", fontsize=8, loc="upper left"
            )
            ax.add_artist(pair_legend)
        if style_handles:
            ax.legend(
                handles=list(style_handles.values()),
                title="Font (marker)",
                fontsize=8,
                loc="lower right",
            )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "semantic_shifts.png"), dpi=300)
    plt.close(fig)

    return stats_df


def run_content_consistency_analysis(
    df,
    style_list,
    original_embeddings,
    num_contents_total,
    num_styles,
    num_contents_sampled,
    seed,
    output_dir,
):
    rng = np.random.default_rng(seed)
    unique_styles = df["style_idx"].unique()
    if len(unique_styles) < num_styles:
        num_styles = len(unique_styles)
    style_choices = rng.choice(unique_styles, size=num_styles, replace=False)

    unique_contents = df["content_idx"].unique()
    contents_to_sample = num_contents_sampled
    if len(unique_contents) < contents_to_sample:
        contents_to_sample = len(unique_contents)
    content_choices = rng.choice(unique_contents, size=contents_to_sample, replace=False)

    subset = df[
        df["style_idx"].isin(style_choices) & df["content_idx"].isin(content_choices)
    ].copy()
    subset["style_name"] = subset["style_idx"].apply(lambda idx: style_list[idx])
    subset["flat_index"] = subset["style_idx"] * num_contents_total + subset["content_idx"]

    intra_dists = []
    content_groups = list(subset.groupby("content_char"))
    for _, group in content_groups:
        indices = group["flat_index"].astype(int).values
        vectors = original_embeddings[indices]
        if vectors.shape[0] < 2:
            continue
        for i in range(vectors.shape[0]):
            for j in range(i + 1, vectors.shape[0]):
                intra_dists.append(float(np.linalg.norm(vectors[i] - vectors[j])))

    inter_dists = []
    for (_, group_a), (_, group_b) in combinations(content_groups, 2):
        indices_a = group_a["flat_index"].astype(int).values
        indices_b = group_b["flat_index"].astype(int).values
        vectors_a = original_embeddings[indices_a]
        vectors_b = original_embeddings[indices_b]
        for point_a in vectors_a:
            for point_b in vectors_b:
                inter_dists.append(float(np.linalg.norm(point_a - point_b)))

    intra_mean = float(np.mean(intra_dists)) if intra_dists else np.nan
    intra_std = float(np.std(intra_dists)) if intra_dists else np.nan
    inter_mean = float(np.mean(inter_dists)) if inter_dists else np.nan
    inter_std = float(np.std(inter_dists)) if inter_dists else np.nan
    separation_score = (
        inter_mean / intra_mean if intra_mean and inter_mean and intra_mean > 0 else np.nan
    )

    print("\n[Task 3] Content consistency metrics (computed in original 1024D space):")
    print(
        f"Intra-class mean={intra_mean:.4f} (std={intra_std:.4f}), "
        f"Inter-class mean={inter_mean:.4f} (std={inter_std:.4f}), "
        f"Separation score (inter/intra)={separation_score:.4f}"
    )

    # Visualization
    contents = sorted(subset["content_char"].unique())
    styles = sorted(subset["style_idx"].unique())
    cmap = matplotlib.colormaps.get_cmap("tab20")
    content_colors = [
        cmap(i / max(1, len(contents) - 1)) for i in range(len(contents))
    ]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(7, 6))
    content_handles = {}
    style_handles = {}
    for _, row in subset.iterrows():
        content_idx = contents.index(row["content_char"])
        style_idx = styles.index(row["style_idx"])
        color = content_colors[content_idx]
        marker = markers[style_idx % len(markers)]
        ax.scatter(row["x"], row["y"], color=color, marker=marker, s=70, alpha=0.85)
        if row["content_char"] not in content_handles:
            content_handles[row["content_char"]] = plt.Line2D(
                [0], [0], color=color, marker="o", linestyle="", label=row["content_char"]
            )
        style_label = f"{row['style_name']} (#{row['style_idx']})"
        if style_label not in style_handles:
            style_handles[style_label] = plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle="",
                label=style_label,
            )

    ax.set_title(
        f"Content clustering ({len(styles)} styles × {len(contents)} contents)"
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    content_legend = ax.legend(
        handles=list(content_handles.values()),
        title="Content",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    ax.add_artist(content_legend)
    ax.legend(
        handles=list(style_handles.values()),
        title="Style",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "content_consistency.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

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
        "selected_content_chars": contents,
        "embedding_space": "original_1024d",
    }
    metrics_path = os.path.join(output_dir, "content_consistency_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, subset


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

    df, lookup = build_dataframe(style_list, content_list, coords_2d)

    # Task 1
    run_semantic_shift_analysis(df, lookup, style_list, args.vector_plot_styles, args.output_dir)

    # Task 3
    run_content_consistency_analysis(
        df,
        style_list,
        codebook,
        len(content_list),
        args.task3_num_styles,
        args.task3_num_contents,
        args.task3_seed,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
