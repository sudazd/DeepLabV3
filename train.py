"""
训练 / 测试 / 复杂度统计脚本（毕节滑坡数据集，7:2:1 划分）

用法示例：
  # 完整模型，3 次不同训练种子（划分种子固定为 42）
  for s in 0 1 2; do python train.py --root /data/Bijie --seed $s --out runs/full_s$s; done
  # 消融：baseline
  python train.py --root /data/Bijie --no_ldaspp --no_gam --no_mrff --out runs/base
  # 复杂度
  python train.py --complexity
"""
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import LandslideDeepLab

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ------------------------------ 数据 ------------------------------
def list_pairs(root):
    """
    默认目录：<root>/landslide/image/*.png  与  <root>/landslide/mask/*.png（同名）。
    只使用含滑坡的 770 张；若你的目录结构不同，请改这里。
    """
    img_dir, msk_dir = Path(root) / "landslide" / "image", Path(root) / "landslide" / "mask"
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXT:
            m = next((msk_dir / (p.stem + e) for e in IMG_EXT if (msk_dir / (p.stem + e)).exists()), None)
            if m is not None:
                pairs.append((str(p), str(m)))
    return pairs


def split_pairs(pairs, split_seed=42, ratio=(0.7, 0.2, 0.1)):
    pairs = pairs[:]
    random.Random(split_seed).shuffle(pairs)
    n = len(pairs)
    a, b = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
    return pairs[:a], pairs[a:b], pairs[b:]


class SegDS(Dataset):
    def __init__(self, pairs, size=512, train=False):
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((self.size, self.size), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk) > 127).astype(np.int64)
        if self.train:  # [ASSUMPTION] 论文未写增强细节；仅对训练集做翻转/旋转/亮度扰动
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5: img, msk = img[::-1], msk[::-1]
            k = random.randint(0, 3)
            img, msk = np.rot90(img, k), np.rot90(msk, k)
            img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
        img = (img - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(msk)))


# ------------------------------ 损失 ------------------------------
class DiceFocal(nn.Module):
    """Dice Loss + Focal Loss（论文未给权重，这里 1:1）[ASSUMPTION]"""

    def __init__(self, gamma=2.0, w_dice=1.0, w_focal=1.0):
        super().__init__()
        self.gamma, self.wd, self.wf = gamma, w_dice, w_focal

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        p = logp.exp()
        pt = p.gather(1, y[:, None]).squeeze(1)
        focal = (-(1 - pt) ** self.gamma * logp.gather(1, y[:, None]).squeeze(1)).mean()
        oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
        inter = (p * oh).sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)).mean()
        return self.wd * dice + self.wf * focal


# ------------------------------ 指标 ------------------------------
class Meter:
    def __init__(self, n=2):
        self.n, self.cm = n, torch.zeros(n, n, dtype=torch.long)

    def update(self, pred, y):
        idx = (y.view(-1) * self.n + pred.view(-1)).cpu()
        self.cm += torch.bincount(idx, minlength=self.n ** 2).view(self.n, self.n)

    def compute(self):
        cm = self.cm.double()
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1)
        prec, rec = tp / (tp + fp).clamp(min=1), tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
        return {
            "mIoU": iou.mean().item() * 100,                  # 背景类与滑坡类 IoU 平均（表1）
            "IoU_landslide": iou[1].item() * 100,             # 滑坡前景 IoU（表3）
            "Precision_fg": prec[1].item() * 100, "Recall_fg": rec[1].item() * 100, "F1_fg": f1[1].item() * 100,
            "Precision_macro": prec.mean().item() * 100,      # 表1 的 P/R/F1 疑似为两类宏平均，见说明
            "Recall_macro": rec.mean().item() * 100, "F1_macro": f1.mean().item() * 100,
            "Accuracy": (tp.sum() / cm.sum()).item() * 100,
        }


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    m = Meter()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        m.update(model(x).argmax(1), y)
    return m.compute()


# ------------------------------ 复杂度 ------------------------------
def complexity(args, dev):
    from torch.utils.flop_counter import FlopCounterMode
    for name, kw in [("DeepLabV3+(baseline)", dict(use_ldaspp=False, use_gam=False, use_mrff=False)),
                     ("Proposed", {})]:
        m = LandslideDeepLab(pretrained=False, **kw).to(dev).eval()
        x = torch.randn(1, 3, 512, 512, device=dev)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        gmacs = fc.get_total_flops() / 2 / 1e9  # 论文的 FLOPs/G 多半是 MACs；grid_sample 不计入
        with torch.no_grad():
            for _ in range(20): m(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            t = time.time()
            for _ in range(100): m(x)
            if dev.type == "cuda": torch.cuda.synchronize()
        print(f"{name:22s} Params {params:.2f}M  GMACs {gmacs:.2f}  Time {(time.time()-t)*10:.2f} ms")


# ------------------------------ 主流程 ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\BijieData\Bijie-landslide-dataset")
    ap.add_argument("--out", default=r"D:\BijieData\DeepLabV3\runs\run100_256")
    ap.add_argument("--epochs", type=int, default=100)      # [ASSUMPTION] 论文未给——训练轮数
    ap.add_argument("--bs", type=int, default=8)            # [ASSUMPTION] 论文未给
    ap.add_argument("--lr", type=float, default=3e-5)       # ——学习率，也需要修改可以让骨干用 3e-5，其他部分用 1e-4 到 3e-4。论文：Adam, 初始 3e-5, 余弦退火
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)          # 训练随机种子
    ap.add_argument("--split_seed", type=int, default=42)   # 数据划分种子（3次重复间保持不变）
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_ldaspp", action="store_true")
    ap.add_argument("--no_gam", action="store_true")
    ap.add_argument("--no_mrff", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--complexity", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.complexity:
        return complexity(args, dev)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tr, va, te = split_pairs(list_pairs(args.root), args.split_seed)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    mk = lambda p, t, sh: DataLoader(SegDS(p, args.size, t), args.bs if t else 1, shuffle=sh,
                                     num_workers=args.workers, drop_last=t, pin_memory=True)
    dl_tr, dl_va, dl_te = mk(tr, True, True), mk(va, False, False), mk(te, False, False)

    model = LandslideDeepLab(2, not args.no_pretrained, not args.no_ldaspp,
                             not args.no_gam, not args.no_mrff).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = DiceFocal()
    best, ckpt = -1, os.path.join(args.out, "best.pth")

    for ep in range(args.epochs):
        model.train()
        tot = 0
        '''for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item()
        '''
        t0 = time.time()
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev), y.to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True);
            loss.backward();
            opt.step()
            tot += loss.item()
            if (i + 1) % 10 == 0:
                print(f"  step {i + 1}/{len(dl_tr)}  {time.time() - t0:.0f}s", flush=True)
        sched.step()
        r = evaluate(model, dl_va, dev)
        if r["mIoU"] > best:  # 以验证集 mIoU 选模型；测试集只在最后用一次
            best = r["mIoU"]; torch.save(model.state_dict(), ckpt)
        print(f"ep{ep+1:03d} loss {tot/len(dl_tr):.4f} val mIoU {r['mIoU']:.2f} "
              f"IoU_ls {r['IoU_landslide']:.2f} (best {best:.2f})", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = evaluate(model, dl_te, dev)
    print("TEST:", json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.out, "test_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
