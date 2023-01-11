"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0

Example:
    python predict.py --config=configs/finetune_funsd_el_spade.yaml --pretrained_model_file=finetune_funsd_el_spade__bros-base-uncased/checkpoints/epoch=49-last.pt

"""
import os
import copy
import cv2
import json
import itertools
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from lightning_modules.bros_bies_module import get_label_map
from lightning_modules.data_modules.bros_dataset import BROSDataset
from model import get_model
from utils import get_class_names, get_config

cfg = get_config()

def load_json_examples(json_path):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    return data

def load_model_weight(net, pretrained_model_file):
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")[
        "state_dict"
    ]
    new_state_dict = {}
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        new_state_dict[new_k] = v
    net.load_state_dict(new_state_dict)


def get_eval_kwargs_bies(dataset_root_path):
    ignore_index = -100
    label_map = get_label_map(dataset_root_path)

    eval_kwargs = {
        "ignore_index": ignore_index,
        "label_map": label_map,
    }

    return eval_kwargs


def get_eval_kwargs_bio(dataset_root_path):
    class_names = get_class_names(dataset_root_path)

    eval_kwargs = {"class_names": class_names}

    return eval_kwargs


def get_eval_kwargs_spade(dataset_root_path, max_seq_length):
    class_names = get_class_names(dataset_root_path)
    dummy_idx = max_seq_length

    eval_kwargs = {"class_names": class_names, "dummy_idx": dummy_idx}

    return eval_kwargs


def get_eval_kwargs_spade_rel(max_seq_length):
    dummy_idx = max_seq_length

    eval_kwargs = {"dummy_idx": dummy_idx}

    return eval_kwargs

def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path, "class_names.txt")
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    class_idx_dic = dict([(class_name, idx) for idx, class_name in enumerate(class_names)])
    return class_names, class_idx_dic


def process_spade(data, backbone_type, tokenizer, class_names, class_idx_dic):
    max_seq_length=512
    pad_token_id = tokenizer.vocab["[PAD]"]
    cls_token_id = tokenizer.vocab["[CLS]"]
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]
    width = data["meta"]["imageSize"]["width"]
    height = data["meta"]["imageSize"]["height"]

    input_ids = np.ones(max_seq_length, dtype=int) * pad_token_id
    bbox = np.zeros((max_seq_length, 8), dtype=np.float32)
    attention_mask = np.zeros(max_seq_length, dtype=int)

    are_box_first_tokens = np.zeros(max_seq_length, dtype=np.bool_)
    el_labels = np.ones(max_seq_length, dtype=int) * max_seq_length

    list_tokens = []
    list_bbs = []
    box2token_span_map = []

    box_to_token_indices = []
    cum_token_idx = 0

    cls_bbs = [0.0] * 8

    for word_idx, word in enumerate(data["words"]):
        this_box_token_indices = []

        tokens = word["tokens"]
        bb = word["boundingBox"]
        if len(tokens) == 0:
            tokens.append(unk_token_id)

        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break

        box2token_span_map.append(
            [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
        )  # including st_idx
        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx)

        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)

    sep_bbs = [width, height] * 4

    # For [CLS] and [SEP]
    list_tokens = (
        [cls_token_id]
        + list_tokens[: max_seq_length - 2]
        + [sep_token_id]
    )
    if len(list_bbs) == 0:
        # When len(data["words"]) == 0 (no OCR result)
        list_bbs = [cls_bbs] + [sep_bbs]
    else:  # len(list_bbs) > 0
        list_bbs = [cls_bbs] + list_bbs[: max_seq_length - 2] + [sep_bbs]

    len_list_tokens = len(list_tokens)
    input_ids[:len_list_tokens] = list_tokens
    attention_mask[:len_list_tokens] = 1

    bbox[:len_list_tokens, :] = list_bbs

    # bounding box normalization -> [0, 1]
    
    bbox_orig = copy.deepcopy(bbox)


    bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
    bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height
    # print(backbone_type)
    if backbone_type == "layoutlm":
        bbox = bbox[:, [0, 1, 4, 5]]
        bbox = bbox * 1000
        bbox = bbox.astype(int)

    st_indices = [
        indices[0]
        for indices in box_to_token_indices
        if indices[0] < max_seq_length
    ]
    are_box_first_tokens[st_indices] = True

    # Label
    relations = data["parse"]["relations"]
    # print("============= relation =============")
    # print(relations)
    for relation in relations:
        if relation[0] >= len(box2token_span_map) or relation[1] >= len(
            box2token_span_map
        ):
            continue
        if (
            box2token_span_map[relation[0]][0] >= max_seq_length
            or box2token_span_map[relation[1]][0] >= max_seq_length
        ):
            continue

        word_from = box2token_span_map[relation[0]][0]
        word_to = box2token_span_map[relation[1]][0]
        el_labels[word_to] = word_from
        # print("word to: ", word_to, "word_from", word_from)

    # print("===========", input_ids)
    # print(np.atleast_2d(input_ids))
    # exit()
    # input_ids = torch.from_numpy(input_ids)
    # bbox = torch.from_numpy(bbox)
    # attention_mask = torch.from_numpy(attention_mask)

    # are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
    # el_labels = torch.from_numpy(el_labels)

    input_ids = torch.from_numpy(input_ids)
    bbox = torch.from_numpy(bbox)
    attention_mask = torch.from_numpy(attention_mask)

    are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
    el_labels = torch.from_numpy(el_labels)

    return_dict = {
        "input_ids":input_ids.unsqueeze(0),
        "bbox": bbox.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "are_box_first_tokens": are_box_first_tokens.unsqueeze(0),
        "el_labels": el_labels.unsqueeze(0),
        
    }

    return return_dict, bbox_orig

def model_loading():
    net = get_model(cfg)
    load_model_weight(net, cfg.pretrained_model_file)
    net.to("cuda")
    net.eval()
    return net



def main(image, data, net, output_dir, filename="unknown.jpg"):

    if cfg.model.backbone in [
        "naver-clova-ocr/bros-base-uncased",
        "naver-clova-ocr/bros-large-uncased",
    ]:
        backbone_type = "bros"
    elif cfg.model.backbone in [
        "microsoft/layoutlm-base-uncased",
        "microsoft/layoutlm-large-uncased",
    ]:
        backbone_type = "layoutlm"
    else:
        raise ValueError(
            f"Not supported model: self.cfg.model.backbone={cfg.model.backbone}"
        )

    mode = "val"
    # ================= dataset ================

    # print("every think is ok")
    # exit()
    class_names, class_idx_dic = get_class_names(cfg.dataset_root_path)
    # print(class_names)
    process_data , bboxes = process_spade(
        data, 
        backbone_type, 
        net.tokenizer, 
        class_names, 
        class_idx_dic
        )
    # print("=========== processed data ==================")
    # print(process_data)
    # print("====================")
    if cfg.model.head == "bies":
        from lightning_modules.bros_bies_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_bies(cfg.dataset_root_path)
    elif cfg.model.head == "bio":
        from lightning_modules.bros_bio_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_bio(cfg.dataset_root_path)
    elif cfg.model.head == "spade":
        from lightning_modules.bros_spade_module import do_eval_epoch_end, do_eval_step

        eval_kwargs = get_eval_kwargs_spade(
            cfg.dataset_root_path, cfg.train.max_seq_length
        )
    elif cfg.model.head == "spade_rel":
        from lightning_modules.bros_spade_rel_module_el import (
            do_eval_epoch_end,
            do_eval_step,
        )

        eval_kwargs = get_eval_kwargs_spade_rel(cfg.train.max_seq_length)
    else:
        raise ValueError(f"Unknown cfg.config={cfg.config}")

    batch = process_data
    for k in batch.keys():
        # print(k)
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(net.backbone.device)

    with torch.no_grad():
        head_outputs, loss = net(batch)

    step_out = do_eval_step(batch, head_outputs, loss, eval_kwargs)
    # print(step_out)
    loss = step_out["loss"]
    batch_gt_rel = step_out["batch_gt_rel"]
    batch_pr_rel = step_out["batch_pr_rel"]
    gg  = [[bboxes[key], bboxes[value]] for key, value in batch_pr_rel[0]]
    for a,b in gg:
        a,b = np.int32(a), np.int32(b)
        coor_a = [(min(a[::2]),min(a[1::2])),(max(a[::2]),max(a[1::2]))]
        coor_b = [(min(b[::2]),min(b[1::2])),(max(b[::2]),max(b[1::2]))]

        cv2.rectangle(image, coor_a[0], coor_a[1], (0,255,0), 2)
        cv2.rectangle(image, coor_b[0], coor_b[1], (0,255,0), 2)
        # s_point = int((ky2-ky1)/2)
        # e_point = int((vy2-vy1)/2)

        cv2.arrowedLine(image, (coor_a[1][0], coor_a[0][1]+(coor_a[1][1]-coor_a[0][1])//2),\
             (coor_b[0][0], coor_b[0][1]+(coor_b[1][1]-coor_b[0][1])//2), (0,255,0), 2)
    
    # Image.fromarray(image).save('temp.jpg')
    cv2.imwrite(os.path.join(output_dir, filename), image)
 


if __name__ == "__main__":
    import glob
    from tqdm import tqdm

    json_path = "./datasets/funsd_spade/preprocessed"
    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(json_path+"/*")
    print("Model Loading ............  ", end="", flush=True)
    net = model_loading()
    print("Done")
    for i in tqdm(range(len(json_files))):
        json_file = json_files[i]
        data = load_json_examples(json_file)
        image_path = os.path.join("datasets/funsd_spade", data["meta"]["image_path"])
        
        image = cv2.imread(image_path)
        main(image, data, net, output_dir, filename=os.path.basename(image_path))
    # main()
