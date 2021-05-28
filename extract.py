"""
Copy to mmdetection/tools/
"""
import argparse
import os
import os.path as osp

import tqdm
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test

import pickle
import cv2
import numpy as np
import mmcv
import os
import os.path as op
import tqdm


from mmcv import Config, DictAction

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector, build_backbone, build_neck



# async def main():
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use mmdet backbone to extract features')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('dataset_dir', help='annotation file')
    parser.add_argument('--output_dir', type=str, default="features")
    parser.add_argument('--samples_per_gpu', type=int, default=1)
    parser.add_argument('--downsample', type=int, default=0, help="Downsample by 2^(--downsample)")
    args = parser.parse_args()
    config_file = args.config
    device = 'cuda:0'

    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_filepath = op.join(args.output_dir, op.split(args.checkpoint)[-1]+".pkl")
    print(f"Results will be saved to {output_filepath}")
    cfg = Config.fromfile(args.config)
    # build the model and load checkpoint, and the backbone
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # print(model.backbone.conv1.weight[0][0])
    # print(model.backbone.unet.conv.weight[0][0])

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint != "pretrained":
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.backbone  #only backbone

    cfg.data.test["type"] = "tSNE_dummy224"
    cfg.data.test["ann_file"] = op.join(args.dataset_dir,"tsne_dummy_anno.csv")
    cfg.data.test["img_prefix"] = args.dataset_dir
    for d in cfg.data.test["pipeline"]:
        if "img_shape" in d.keys():
            d["img_shape"] = (224, 224)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)


    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    for sample in tqdm.tqdm(data_loader):
        tmp_results = []
        features_batch = model(sample['img'][0])[-1]
        features_batch = features_batch.cpu().detach().numpy()
        for idx, features in enumerate(features_batch): 
            img_metas = {}
            img_metas_dc = sample['img_metas'][0]._data[0][idx]
            for key in img_metas_dc.keys():
                img_metas[key] = img_metas_dc[key]
            tmp_results.append({"features": features, "img_metas":img_metas})
        results.extend(tmp_results)
    with open(output_filepath, "wb") as handle:
        pickle.dump(results, handle)
    print(output_filepath)