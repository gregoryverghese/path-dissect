import os
import argparse
import datetime
import json
import pandas as pd
import torch

from path_dissect import similarity
from path_dissect.utils import (
    save_activations,
    get_save_names,
    get_similarity_from_activations,
    save_cem_activations,
    save_plip_slide_features,
    save_clip_text_features,
)
from path_dissect.vlms import load_vlm
from path_dissect.datasets import (
    SlideEmbeddingDataset,
    UNI_EMB_DIR,
    PLIP_EMB_DIR,
    CEM_CHECKPOINT,
    get_cem_model,
)


parser = argparse.ArgumentParser(description='path-dissect: concept-based neuron description')

parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                             'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'plip', 'conch'],
                    help="Which VLM to use for concept grounding")
parser.add_argument("--conch_checkpoint", type=str, default=None,
                    help="Path to CONCH model weights (required when --clip_model conch)")
parser.add_argument("--target_model", type=str, default="resnet50",
                    help="""Which model to dissect, supported options are pretrained imagenet models from
                         torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces).
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="broden",
                    choices=["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "tcga"])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi",
                    choices=["soft_wpmi", "wpmi", "rank_reorder", "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")

    similarity_fn = eval("similarity.{}".format(args.similarity_fn))

    vlm_kwargs = {}
    if args.clip_model == "conch":
        if args.conch_checkpoint is None:
            raise ValueError("--conch_checkpoint is required when using --clip_model conch")
        vlm_kwargs["conch_checkpoint"] = args.conch_checkpoint

    with open(args.concept_set, 'r') as f:
        words = [w for w in (f.read()).split('\n') if w != ""]

    # ------------------------------------------------------------------ #
    # CEM pathway — slide-level MIL model                                 #
    # ------------------------------------------------------------------ #
    if args.target_model == "cem":
        slide_dataset = SlideEmbeddingDataset(UNI_EMB_DIR)
        slide_ids = slide_dataset.slide_ids

        # Save names
        concept_set_name = (args.concept_set.split("/")[-1]).split(".")[0]
        vlm_name = args.clip_model.replace("/", "")
        target_save_name = f"{args.activation_dir}/cem_concept_bottleneck.pt"
        clip_save_name   = f"{args.activation_dir}/tcga_{vlm_name}_slides.pt"
        text_save_name   = f"{args.activation_dir}/{concept_set_name}_{vlm_name}.pt"

        # CEM activations
        cem_model = get_cem_model(CEM_CHECKPOINT, args.device)
        save_cem_activations(cem_model, slide_dataset, target_save_name, args.device)

        # VLM slide image embeddings (pre-computed per-slide .pt files)
        from path_dissect.datasets import CONCH_EMB_DIR
        emb_dir = CONCH_EMB_DIR if args.clip_model == "conch" else PLIP_EMB_DIR
        save_plip_slide_features(emb_dir, clip_save_name, slide_ids)

        # VLM text embeddings
        vlm = load_vlm(args.clip_model, args.device, **vlm_kwargs)
        text_tokens = vlm.tokenize(words, device=args.device)
        save_clip_text_features(vlm, text_tokens, text_save_name, args.batch_size)

        similarities = get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name,
            similarity_fn, return_target_feats=False, device=args.device
        )
        vals, ids = torch.max(similarities, dim=1)
        del similarities
        torch.cuda.empty_cache()

        descriptions = [words[int(idx)] for idx in ids]
        outputs = {
            "layer": ["concept_bottleneck"] * len(vals),
            "unit": list(range(len(vals))),
            "description": descriptions,
            "similarity": vals.cpu().numpy().tolist(),
        }

    # ------------------------------------------------------------------ #
    # Standard pathway                                                     #
    # ------------------------------------------------------------------ #
    else:
        vlm_kwargs = {}
        if args.clip_model == "conch":
            vlm_kwargs["conch_checkpoint"] = args.conch_checkpoint

        save_activations(
            clip_name=args.clip_model, target_name=args.target_model,
            target_layers=args.target_layers, d_probe=args.d_probe,
            concept_set=args.concept_set, batch_size=args.batch_size,
            device=args.device, pool_mode=args.pool_mode,
            save_dir=args.activation_dir, **vlm_kwargs,
        )

        outputs = {"layer": [], "unit": [], "description": [], "similarity": []}

        for target_layer in args.target_layers:
            save_names = get_save_names(
                clip_name=args.clip_model, target_name=args.target_model,
                target_layer=target_layer, d_probe=args.d_probe,
                concept_set=args.concept_set, pool_mode=args.pool_mode,
                save_dir=args.activation_dir,
            )
            target_save_name, clip_save_name, text_save_name = save_names

            similarities = get_similarity_from_activations(
                target_save_name, clip_save_name, text_save_name,
                similarity_fn, return_target_feats=False, device=args.device
            )
            vals, ids = torch.max(similarities, dim=1)
            del similarities
            torch.cuda.empty_cache()

            descriptions = [words[int(idx)] for idx in ids]
            outputs["unit"].extend(range(len(vals)))
            outputs["layer"].extend([target_layer] * len(vals))
            outputs["description"].extend(descriptions)
            outputs["similarity"].extend(vals.cpu().numpy())

    df = pd.DataFrame(outputs)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    save_path = "{}/{}_{}".format(
        args.result_dir, args.target_model,
        datetime.datetime.now().strftime("%y_%m_%d_%H_%M"),
    )
    os.mkdir(save_path)
    df.to_csv(os.path.join(save_path, "descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
