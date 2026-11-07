"""
用训练好的权重对 train/val/test 集做预测，保存可视化图，并重新计算指标。

每张输出图由 4 个面板横向拼成：
  [原图] [真值叠加(红)] [预测叠加(红)] [误差图：绿=预测正确的滑坡，红=误检(FP)，蓝=漏检(FN)]

用法（Windows 示例，--size 和三个 --no_* 开关必须与训练时一致）：
  python predict.py --root D:\\BijieData\\Bijie-landslide-dataset ^
      --ckpt D:\\BijieData\\DeepLabV3\\runs\\full_256_s0\\best.pth ^
      --size 256 --out_dir D:\\BijieData\\DeepLabV3\\vis\\full_256
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from model import LandslideDeepLab
from train import list_pairs, split_pairs, SegDS, Meter


def overlay(img, m, color=(255, 0, 0), a=0.5):
    out = img.astype(np.float32).copy()
    out[m == 1] = out[m == 1] * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="vis")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--split_seed", type=int, default=42)  # 必须与训练时一致，否则划分不同
    ap.add_argument("--no_ldaspp", action="store_true")
    ap.add_argument("--no_gam", action="store_true")
    ap.add_argument("--no_mrff", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = split_pairs(list_pairs(args.root), args.split_seed)
    pairs = {"train": tr, "val": va, "test": te}[args.split]
    print(f"{args.split}: {len(pairs)} 张")
    ds = SegDS(pairs, args.size, train=False)

    model = LandslideDeepLab(2, False, not args.no_ldaspp, not args.no_gam, not args.no_mrff).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    meter, rows = Meter(), []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            pred = model(x[None].to(dev)).argmax(1)[0].cpu()
            meter.update(pred, y)

            p, g = pred.numpy(), y.numpy()
            tp = int(((p == 1) & (g == 1)).sum())
            fp = int(((p == 1) & (g == 0)).sum())
            fn = int(((p == 0) & (g == 1)).sum())
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

            ip = pairs[i][0]
            stem = os.path.splitext(os.path.basename(ip))[0]
            img = np.asarray(Image.open(ip).convert("RGB").resize((args.size, args.size), Image.BILINEAR))
            err = np.zeros((*p.shape, 3), np.uint8)
            err[(p == 1) & (g == 1)] = (0, 200, 0)
            err[(p == 1) & (g == 0)] = (255, 0, 0)
            err[(p == 0) & (g == 1)] = (0, 120, 255)
            canvas = np.concatenate([img, overlay(img, g), overlay(img, p), err], axis=1)
            Image.fromarray(canvas).save(os.path.join(args.out_dir, f"{i:03d}_{stem}.png"))
            rows.append((stem, iou, tp, fp, fn))

    # 逐张前景 IoU，从差到好排序，方便找出最失败的样本
    rows.sort(key=lambda r: 2 if r[1] != r[1] else r[1])
    with open(os.path.join(args.out_dir, "per_image_iou.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "IoU_landslide", "TP", "FP", "FN"])
        w.writerows(rows)

    res = meter.compute()
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.out_dir, f"{args.split}_metrics_recomputed.json"), "w"), indent=1)
    print("可视化图和 per_image_iou.csv 已保存到:", args.out_dir)


if __name__ == "__main__":
    main()
