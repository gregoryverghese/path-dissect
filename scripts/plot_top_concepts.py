"""
Plot top-N concepts per CEM neuron with ground-truth labels.
Saves top5 and top10 figures to results/.

Usage (legacy PLIP):
    python scripts/plot_top_concepts.py

Usage (specify concept set + VLM):
    python scripts/plot_top_concepts.py \
        --concept_set concept_sets/curated_concepts.txt \
        --vlm conch \
        --tag curated_conch
"""
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dissect import similarity as sim_module
from path_dissect.utils import get_similarity_from_activations

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--concept_set", default="concept_sets/pathology_concepts_combined.txt")
parser.add_argument("--vlm", default="plip", help="plip or conch")
parser.add_argument("--tag", default=None, help="output filename prefix (default: derived from vlm+concept_set)")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

concept_set_name = os.path.splitext(os.path.basename(args.concept_set))[0]
vlm_name = args.vlm.replace("/", "")
tag = args.tag or f"{concept_set_name}_{vlm_name}"

slides_pt  = f"saved_activations/tcga_{vlm_name}_slides.pt"
text_pt    = f"saved_activations/{concept_set_name}_{vlm_name}.pt"

# ── data ──────────────────────────────────────────────────────────────────────
with open(args.concept_set) as f:
    concepts = [l.strip() for l in f if l.strip()]

similarities = get_similarity_from_activations(
    "saved_activations/cem_concept_bottleneck.pt",
    slides_pt,
    text_pt,
    sim_module.soft_wpmi,
    return_target_feats=False,
    device=args.device,
)  # [20, n_concepts]

# neuron index → (short name, ground-truth label, group, subtitle)
NEURONS = [
    ("stage_var_0",      "Stage I",    "Stage",  ""),
    ("stage_var_1",      "Stage II",   "Stage",  ""),
    ("stage_var_2",      "Stage III",  "Stage",  ""),
    ("stage_var_3",      "Stage IV",   "Stage",  ""),
    ("age_group_2",      "Age >70",    "Age",    ""),
    ("age_group_1",      "Age 50-70",  "Age",    ""),
    ("age_group_0",      "Age <50",    "Age",    ""),
    ("project_id_0",     "TCGA-BRCA",  "Cancer", "Breast Invasive Carcinoma"),
    ("project_id_1",     "TCGA-COAD",  "Cancer", "Colon Adenocarcinoma"),
    ("project_id_2",     "TCGA-HNSC",  "Cancer", "Head & Neck Squamous Cell Carcinoma"),
    ("project_id_3",     "TCGA-KICH",  "Cancer", "Kidney Chromophobe"),
    ("project_id_4",     "TCGA-KIRC",  "Cancer", "Kidney Clear Cell Carcinoma"),
    ("project_id_5",     "TCGA-KIRP",  "Cancer", "Kidney Papillary Cell Carcinoma"),
    ("project_id_6",     "TCGA-LUAD",  "Cancer", "Lung Adenocarcinoma"),
    ("project_id_7",     "TCGA-LUSC",  "Cancer", "Lung Squamous Cell Carcinoma"),
    ("project_id_8",     "TCGA-PAAD",  "Cancer", "Pancreatic Adenocarcinoma"),
    ("project_id_9",     "TCGA-READ",  "Cancer", "Rectum Adenocarcinoma"),
    ("RNA_Bio_ter_low",  "RNA low",    "RNA",    ""),
    ("RNA_Bio_ter_mid",  "RNA mid",    "RNA",    ""),
    ("RNA_Bio_ter_high", "RNA high",   "RNA",    ""),
]

GROUP_COLORS = {
    "Stage":  "#4C72B0",
    "Age":    "#55A868",
    "Cancer": "#C44E52",
    "RNA":    "#8172B2",
}


def truncate(s, n=48):
    return s if len(s) <= n else s[:n-1] + "…"


def make_figure(top_n, out_path):
    n_neurons = len(NEURONS)
    ncols = 4
    nrows = (n_neurons + ncols - 1) // ncols  # 5 rows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 7.5, nrows * (top_n * 0.42 + 1.1)),
    )
    axes = axes.flatten()

    for i, (neuron, gt, group, subtitle) in enumerate(NEURONS):
        ax = axes[i]
        vals, ids = torch.topk(similarities[i], top_n)
        vals = vals.cpu().numpy()
        labels = [truncate(concepts[int(idx)]) for idx in ids]

        color = GROUP_COLORS[group]
        # Rank 1 = full opacity, descending
        alphas = np.linspace(0.9, 0.35, top_n)
        colors = [(*matplotlib.colors.to_rgb(color), a) for a in alphas]

        y = np.arange(top_n)
        bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("soft_wpmi", fontsize=7)
        ax.xaxis.set_tick_params(labelsize=7)

        # Title: neuron name + ground truth (+ full cancer name if present)
        title = f"{neuron}  |  GT: {gt}"
        if subtitle:
            title += f"\n{subtitle}"
        ax.set_title(
            title,
            fontsize=9, fontweight="bold",
            color=color, loc="left", pad=4,
        )

        # Thin border in group colour
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=6.5, color="#444",
            )

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower right", ncol=4,
        fontsize=9, framealpha=0.8,
        bbox_to_anchor=(0.98, 0.01),
    )

    fig.suptitle(
        f"CEM concept bottleneck — {vlm_name.upper()} top-{top_n} concepts [{concept_set_name}]\n"
        f"(soft_wpmi · TCGA slides · {len(concepts):,} concepts)",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_summary_figure(out_path):
    """Single chart: one bar per neuron, labelled with top-1 concept."""
    fig, ax = plt.subplots(figsize=(13, 8))

    n = len(NEURONS)
    y = np.arange(n)
    vals, labels, colors, ytick_labels = [], [], [], []

    for i, (neuron, gt, group, subtitle) in enumerate(NEURONS):
        top_val, top_idx = torch.topk(similarities[i], 1)
        top_concept = concepts[int(top_idx[0])]
        v = top_val[0].item()

        vals.append(v)
        labels.append(truncate(top_concept, 60))
        colors.append(GROUP_COLORS[group])

        if subtitle:
            ytick_labels.append(f"{neuron}  |  {gt}\n{subtitle}")
        else:
            ytick_labels.append(f"{neuron}  |  {gt}")

    bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("soft_wpmi similarity", fontsize=9)
    ax.xaxis.set_tick_params(labelsize=8)

    # Concept label on each bar
    for bar, label, v in zip(bars, labels, vals):
        ax.text(
            v + 0.001, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=7.5, color="#222",
        )

    # Extend x-axis to fit labels
    ax.set_xlim(0, max(vals) * 3.2)

    # Group dividers
    boundaries = [0, 4, 7, 17, 20]
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="#aaa", linewidth=0.8, linestyle="--")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.8)

    ax.set_title(
        f"CEM concept bottleneck — {vlm_name.upper()} top-1 concept per neuron [{concept_set_name}]\n"
        f"(soft_wpmi · TCGA slides · {len(concepts):,} concepts)",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_figure(5,  f"results/{tag}_top5.png")
make_figure(10, f"results/{tag}_top10.png")
make_summary_figure(f"results/{tag}_top1_summary.png")
