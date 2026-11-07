"""
一键生成验证集（精度验证）和测试集的预测图与精度指标。
与 model.py、train.py、predict.py 放在同一个文件夹，直接在 PyCharm 里右键运行即可。
结果保存到 D:\\BijieData\\DeepLabV3\\Prediction\\val 和 \\test。
"""
import os
import sys

import predict

# ======== 只需要改这几处 ========
ROOT = r"D:\BijieData\Bijie-landslide-dataset"
CKPT = r"D:\BijieData\DeepLabV3\runs\run100_256\best.pth"  # 改成真实的权重路径（去掉 < >）
SIZE = 256        # 必须与该权重训练时的 --size 一致（256 或 512）
SPLIT_SEED = 42   # 必须与训练时一致
EXTRA = []        # 训练时用过消融开关就写进来，例如 ["--no_gam"] 或 ["--no_ldaspp", "--no_gam", "--no_mrff"]
OUT = r"D:\BijieData\DeepLabV3\Prediction"
# ================================

for split in ("val", "test"):
    sys.argv = [
        "predict.py",
        "--root", ROOT,
        "--ckpt", CKPT,
        "--size", str(SIZE),
        "--split_seed", str(SPLIT_SEED),
        "--split", split,
        "--out_dir", os.path.join(OUT, split),
    ] + EXTRA
    print("=" * 20, split, "=" * 20, flush=True)
    predict.main()