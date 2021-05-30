# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/COCO_R50_epoch_1.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme
# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/ALL_R50.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme
# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/ALL_PUR50.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme

# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/C5S5_R50.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme
# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/C5S5_PUR50.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme
# python tsne.py /tmp2/igor/lowlight-t-SNE/Sony_features/C1S1_PUR50.pth.pkl --json=str_labeled_new_Sony_RX100m7_test.json --output_dir=tsne --binary_label=isextreme

# https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
import argparse
from re import L
import tqdm
from PIL import Image
import glob
import os
import os.path as op
import pickle
import tqdm

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib.lines import Line2D

import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from pycocotools.coco import COCO
import json

TSNE_DEFAULT = {"n_iter" : 3*5000, "random_state" : 3}

def scatter(x, colors, cats, alpha=1):
    # https://github.com/oreillymedia/t-SNE-tutorial
    # We choose a color palette with seaborn.
    rgb_palette = np.array(sns.color_palette("hls", len(cats)))
    palette = np.c_[rgb_palette, alpha*np.ones(rgb_palette.shape[0])]

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each class.
    txts = []
    for i in range(len(cats)):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    # custom legend
    # rgba_palette = x=np.c_[palette, np.ones(palette.shape[0])]
    legend_elements = [Line2D([0], [0], marker='o', color=palette[i], label="{}, ({})".format(cats[i], i),
                          markerfacecolor=palette[i], markersize=15) for i in range(len(cats))]

    ax.legend(handles=legend_elements, loc='lower right')
    return f, ax, sc, txts

def read_pickle(args):
    x = []
    y = []
    with open(args.pickled_features, "rb") as handle:
        data = pickle.load(handle)
        for idx, sample in tqdm.tqdm(enumerate(data)):
            features = sample["features"]
            img_metas = sample["img_metas"]
            x.append(features)
            y.append(img_metas)
    x = np.array(x)
    B, C, H, W = x.shape
    std=np.moveaxis(x, 0, 1).std(axis=(1,2,3))
    for c in range(C):
        x[:, c, :, :]/=std[c] 
    x = x.reshape(B, -1)
    return x, y


if __name__ == "__main__":
                
    # https://github.com/oreillymedia/t-SNE-tutorial
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pickled_features')
    parser.add_argument('--json', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lr', default=200, type=int)
    parser.add_argument('--perplexity', default=None, type=int)
    parser.add_argument('--binary_label', default=None)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--cats', action="store_true")
    parser.add_argument('--cat', default=None, type=str)
    args = parser.parse_args()

    assert 1.0 <= args.lr <= 1000.0
    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    plot_dir = op.join(args.output_dir, op.split(args.pickled_features)[-1].split(".")[0])
    if not op.exists(plot_dir):
        os.mkdir(plot_dir)
    assert op.exists(plot_dir)

    assert sum([bool(arg)*1 for arg in [args.binary_label, args.labels, args.cats, args.cat]]) == 1

    with open(args.json) as json_file:
        cocoGt_data = json.load(json_file)
    cocoGt = COCO(args.json)

    dir, filename = op.split(args.pickled_features)
    dist_fp = op.join(dir, "dist_"+filename.split(".")[0]+".npy")
    y_fp = op.join(dir, "y_"+filename.split(".")[0]+".npy")
    if not (op.exists(dist_fp) and op.exists(y_fp)):
        x, y = read_pickle(args)
        print("Calculating distances...")
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        pca = PCA(n_components=2, svd_solver='randomized',
                      random_state=3)
        X_embedded = pca.fit_transform(x).astype(np.float32, copy=False)
        dist = pairwise_distances(X_embedded)
        np.save(dist_fp, dist)
        np.save(y_fp, y)
    else:
        print("Loading in distances...")
        dist = np.load(dist_fp)
        y = np.load(y_fp, allow_pickle=True)

    labels = []
    perplexities = [args.perplexity] if args.perplexity else [10, 30, 50]
    if any([args.labels, args.cat]):
        raise NotImplementedError
    if args.binary_label or args.cats:
        if args.binary_label:
            prefix = args.binary_label
            for metadata in y:
                ann_id = int(metadata["ori_filename"].split(".")[0])
                ann = cocoGt.loadAnns([ann_id])[0]
                label = ann[args.binary_label]
                labels.append(label)
            # convert booleans to text
            label_name = args.binary_label.split("is")[-1]
            labels = [label_name if label else f"not {label_name}" for label in labels]
        if args.cats:
            prefix = "cats"
            for metadata in y:
                ann_id = int(metadata["ori_filename"].split(".")[0])
                ann = cocoGt.loadAnns([ann_id])[0]
                label = cocoGt.loadCats(ids=[ann["category_id"]])[0]["name"]
                labels.append(label)
        labels_unique = list(set(labels))
        labels_unique.sort()
        labels_dict = {label : idx for idx, label in enumerate(labels_unique)}
        labels_legend = [label for label in labels_unique]
        labels_int = [labels_dict[label] for label in labels]

    assert labels
    for perplexity in perplexities:
        tsne = TSNE(learning_rate=args.lr, perplexity=perplexity, metric="precomputed", verbose=True, **TSNE_DEFAULT).fit_transform(dist)
        scatter(tsne, np.array(labels_int), labels_legend, alpha=0.5)
        print(op.join(plot_dir, f"{prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{prefix}_p{perplexity}_lr{int(args.lr)}.svg"))
    