import os
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import data_utils
from vlm_wrapper import load_vlm

PM_SUFFIX = {"max":"_max", "avg":""}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    """
    model: VLMWrapper — encode_text accepts batches of whatever tokenize() returned.
    text: output of model.tokenize() — either a tensor (CLIP) or a dict of tensors (PLIP/CONCH).
    """
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    n = text.shape[0] if isinstance(text, torch.Tensor) else len(next(iter(text.values())))
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(n / batch_size))):
            batch = (text[batch_size*i:batch_size*(i+1)] if isinstance(text, torch.Tensor)
                     else {k: v[batch_size*i:batch_size*(i+1)] for k, v in text.items()})
            text_features.append(model.encode_text(batch))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    n = text.shape[0] if isinstance(text, torch.Tensor) else len(next(iter(text.values())))
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(n / batch_size))):
            batch = (text[batch_size*i:batch_size*(i+1)] if isinstance(text, torch.Tensor)
                     else {k: v[batch_size*i:batch_size*(i+1)] for k, v in text.items()})
            text_features.append(model.encode_text(batch))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, target_name, target_layers, d_probe,
                     concept_set, batch_size, device, pool_mode, save_dir, **vlm_kwargs):

    vlm = load_vlm(clip_name, device, **vlm_kwargs)
    target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    # setup data
    data_c = data_utils.get_data(d_probe, vlm.preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f:
        words = (f.read()).split('\n')
    words = [w for w in words if w != ""]

    text = vlm.tokenize(words, device=device)

    save_names = get_save_names(clip_name=clip_name, target_name=target_name,
                                target_layer='{}', d_probe=d_probe, concept_set=concept_set,
                                pool_mode=pool_mode, save_dir=save_dir)
    target_save_name, clip_save_name, text_save_name = save_names

    save_clip_text_features(vlm, text, text_save_name, batch_size)
    save_clip_image_features(vlm, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, vlm, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    vlm: VLMWrapper instance
    """
    pred_tokens = vlm.tokenize(preds, device=device)
    gt_tokens = vlm.tokenize(gt, device=device)
    pred_embeds = []
    gt_embeds = []

    n = (pred_tokens.shape[0] if isinstance(pred_tokens, torch.Tensor)
         else len(next(iter(pred_tokens.values()))))
    with torch.no_grad():
        for i in range(math.ceil(n / batch_size)):
            def _batch(t, i):
                if isinstance(t, torch.Tensor):
                    return t[batch_size*i:batch_size*(i+1)]
                return {k: v[batch_size*i:batch_size*(i+1)] for k, v in t.items()}
            pred_embeds.append(vlm.encode_text(_batch(pred_tokens, i)))
            gt_embeds.append(vlm.encode_text(_batch(gt_tokens, i)))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

    
    