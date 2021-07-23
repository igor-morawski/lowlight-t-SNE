"""
Resize while keeping aspect ratio 
# https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
"""

import argparse
import tqdm
from PIL import Image
import glob
import os
import os.path as op
import json
from pycocotools.coco import COCO

def make_square(im, min_size, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# python prepare.py --json=str_labeled_new_Sony_RX100m7_test.json --jpg_dir=/tmp2/igor/Dataset --output_dir=Sony --size=224
# python prepare.py --json=str_labeled_new_Sony_RX100m7_test.json --jpg_dir=/tmp2/igor/Dataset --output_dir=Sony_small --size=224 --occupy_half
# python prepare.py --json=str_labeled_new_Sony_RX100m7_test.json --jpg_dir=/tmp2/igor/Dataset --output_dir=Sony_label --size=224 --label --cat=person

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', type=str)
    parser.add_argument('--jpg_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--extension', default="jpg", help='File extension, e.g. "png"')
    parser.add_argument('--size', type=int)
    parser.add_argument('--occupy_half', action="store_true")
    parser.add_argument('--label', action="store_true")
    parser.add_argument('--cat', default=None, type=str)
    args = parser.parse_args()
    paths = []

    with open(args.json) as json_file:
        data = json.load(json_file)
    cocoGt = COCO(args.json)

    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    
    lines = []
    for ann in tqdm.tqdm(data['annotations']):
        ann_id = ann['id']
        x1, y1, W, H = ann['bbox']
        x2, y2 = x1 + W, y1 + H
        img_id = ann['image_id']
        img_info = cocoGt.loadImgs([img_id])
        assert len(img_info) == 1
        img_fp = op.join(args.jpg_dir, img_info[0]['file_name'])
        cat = cocoGt.loadCats(ann["category_id"])[0]["name"]
        assert op.exists(img_fp)
        file_name = f"{ann_id}.{args.extension}"
        if args.label: 
            isextreme = ann["isextreme"]
            file_name = f'{"1" if isextreme else "0"}_' + file_name
        out_file_name = op.join(args.output_dir, file_name)
        if args.cat:
            file_name = f'{cat}_' + file_name
        if args.cat and (cat != args.cat):
            continue
        if not op.exists(out_file_name):
            img = Image.open(img_fp)
            img = img.crop((x1,y1,x2,y2))
            img = make_square(img, min_size=args.size).resize((args.size,args.size))
            if args.occupy_half:
                canvas = Image.new('RGB', img.size)
                img = img.resize([s//2 for s in img.size])
                canvas.paste(img, [s//2 for s in img.size])
                img = canvas
            img.save(out_file_name)
            assert img.size == (args.size, args.size)
        line = f"{file_name},1,2,3,4,{cat}"
        lines.append(line)
    with open(op.join(args.output_dir, "tsne_dummy_anno.csv"), 'w') as f:
        text = "\n".join(lines)+"\n"
        f.write(text)