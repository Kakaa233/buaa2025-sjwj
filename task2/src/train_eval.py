from pathlib import Path
import json
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import torchvision.transforms as T

# 确保从仓库根目录运行时，本地模块导入正常
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import AnomalyDataset
from model import FeatureExtractor, GaussianModel


# 可调的配置 

CATEGORIES = ['hazelnut', 'zipper']
IMAGE_SIZE = 320           # 提升输入分辨率以保留小缺陷细节（如榛子蛀孔、拉链边缘毛边）
USE_PCA = True             # 启用 PCA 可降低噪声、稳定协方差估计
PCA_DIM = 128              # 常用 64/128，结合数据规模与效果权衡
SCORE_MODE = 'two-class'   # 'two-class': lp_bad - lp_good；'one-class': -lp_good（仅好类密度）

USE_PATCH = False          # hazelnut 用滑窗/patch聚合（对小孔/裂痕更敏感）
PATCH_SIZE = 96            # patch 尺寸（像素，方形，需 < IMAGE_SIZE）
PATCH_STRIDE = 64          # patch 步长（像素）
PATCH_AGG = 'max'          # 'max' 或 'topk_mean'
PATCH_TOPK = 3             # top-k 聚合时的 k

USE_ZIPPER_ROI = False     # zipper 仅在条带ROI评分（沿拉链方向的窄条）
ROI_RATIO = 0.35           # ROI 相对宽度（0-1），例如 0.35 表示取中间35%宽度条带


def build_base_transform(image_size: int):
    """与数据集一致的预处理，用于单张图像/patch的特征提取。"""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def score_images(model: GaussianModel, extractor: FeatureExtractor, device: torch.device,
                 paths: list, image_size: int,
                 use_patch: bool, patch_size: int, patch_stride: int, patch_agg: str, patch_topk: int,
                 use_roi: bool, roi_ratio: float,
                 scaler: StandardScaler, pca: PCA | None,
                 score_mode: str) -> np.ndarray:
    """对给定图像路径列表打分，支持：
    - 滑窗/patch 聚合：对每张图提取多个 patch 的分数，聚合为图像分数
    - zipper ROI：对齐取中间竖向条带，仅在条带内评分
    返回每张图像的最终分数数组。
    """
    transform = build_base_transform(image_size)

    def img_to_tensor(img: Image.Image) -> torch.Tensor:
        return transform(img).unsqueeze(0).to(device)

    def score_tensor_batch(t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = extractor(t)
            # 标准化/PCA
            X_np = f.cpu().numpy()
            X_np = scaler.transform(X_np)
            if pca is not None:
                X_np = pca.transform(X_np)
            X_t = torch.from_numpy(X_np).float()
            if score_mode == 'two-class':
                s = model.anomaly_score(X_t)
            else:
                s = -model.log_prob(X_t, good=True)
            return s

    scores = []
    for p in tqdm(paths, desc='ScoreImages'):
        img = Image.open(p).convert('RGB')
        W, H = img.size

        # 计算 ROI（zipper）
        if use_roi:
            roi_w = max(8, int(W * roi_ratio))
            x0 = (W - roi_w) // 2
            img_roi = img.crop((x0, 0, x0 + roi_w, H))
            img_use = img_roi
        else:
            img_use = img

        # 不使用 patch：整图一次评分
        if not use_patch:
            t = img_to_tensor(img_use)
            s = score_tensor_batch(t)
            scores.append(float(s.item()))
            continue

        # 使用 patch：在预裁剪后的图上滑窗
        # 为避免与 transform 冲突，这里在原图上切 patch，再对每个 patch 应用相同的 transform
        patches = []
        W2, H2 = img_use.size
        ps = min(patch_size, min(W2, H2))
        stride = max(8, min(patch_stride, ps))
        for y in range(0, H2 - ps + 1, stride):
            for x in range(0, W2 - ps + 1, stride):
                patch = img_use.crop((x, y, x + ps, y + ps))
                patches.append(patch)
        if not patches:
            patches = [img_use]
        # 批处理提速：分批堆叠
        batch_tensors = []
        for ph in patches:
            batch_tensors.append(transform(ph))
        batch = torch.stack(batch_tensors, dim=0).to(device)
        s_batch = score_tensor_batch(batch).cpu().numpy()

        # 聚合策略
        if patch_agg == 'max':
            s_final = float(s_batch.max())
        elif patch_agg == 'topk_mean':
            k = max(1, min(patch_topk, len(s_batch)))
            s_final = float(np.sort(s_batch)[-k:].mean())
        else:
            s_final = float(s_batch.mean())
        scores.append(s_final)

    return np.array(scores)


def extract_features(dataloader, extractor, device):
    """提取批量特征并拼接到 CPU 张量。
    返回: (features[B, D], labels[B])
    """
    feats, labels = [], []
    with torch.no_grad():
        for imgs, ys, _ in tqdm(dataloader, desc='Extract'):
            imgs = imgs.to(device)
            f = extractor(imgs)
            feats.append(f.cpu())
            labels.append(ys)
    return torch.cat(feats), torch.cat(labels)


def train_and_eval(data_root: str):
    root = Path(data_root)
    json_path = root / 'image_anomaly_labels.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = FeatureExtractor().to(device)

    metrics_all: Dict[str, Dict[str, float]] = {}

    for category in CATEGORIES:
        # 1) 加载数据：提高分辨率，轻微有助于检测小型缺陷
        train_ds = AnomalyDataset(root, category, 'train', json_path, image_size=IMAGE_SIZE)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)

        # 2) 提取训练特征
        X_train, y_train = extract_features(train_loader, extractor, device)

        # 3) 特征标准化 + 可选 PCA 降维（在训练集上拟合变换，再用于测试集），
        #    目标：
        #    标准化减少通道尺度差异
        #    降维降低噪声，稳定后续高斯协方差估计
        X_train_np = X_train.numpy()
        scaler = StandardScaler().fit(X_train_np)
        X_train_np = scaler.transform(X_train_np)

        pca = None
        if USE_PCA:
            # 不使用 whiten=True，避免放大噪声；随机种子固定便于复现
            pca = PCA(n_components=PCA_DIM, svd_solver='auto', whiten=False, random_state=0)
            X_train_np = pca.fit_transform(X_train_np)
        X_train_t = torch.from_numpy(X_train_np).float()

        # 4) 拟合高斯密度（监督版：好/坏各一）
        model = GaussianModel(dim=X_train_t.size(1))
        model.fit(X_train_t, y_train)

        # 5) 测试集：若启用 patch/ROI，则基于原图路径按策略打分；否则走批量特征路径
        test_ds = AnomalyDataset(root, category, 'test', json_path, image_size=IMAGE_SIZE)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        if USE_PATCH or (USE_ZIPPER_ROI and category == 'zipper'):
            # 直接基于原图路径评分（包含预处理、特征提取、标准化/PCA、打分与聚合）
            test_paths = [p for _, _, p in test_ds]
            y_test = torch.tensor([label for _, label, _ in test_ds], dtype=torch.int64)
            s = score_images(model, extractor, device,
                             test_paths, IMAGE_SIZE,
                             USE_PATCH, PATCH_SIZE, PATCH_STRIDE, PATCH_AGG, PATCH_TOPK,
                             use_roi=(USE_ZIPPER_ROI and category == 'zipper'), roi_ratio=ROI_RATIO,
                             scaler=scaler, pca=pca,
                             score_mode=SCORE_MODE)
        else:
            # 原始整图批量特征路径
            X_test, y_test = extract_features(test_loader, extractor, device)
            with torch.no_grad():
                X_test_np = scaler.transform(X_test.numpy())
                if pca is not None:
                    X_test_np = pca.transform(X_test_np)
                X_test_t = torch.from_numpy(X_test_np).float()
                if SCORE_MODE == 'two-class':
                    scores = model.anomaly_score(X_test_t)
                else:
                    scores = -model.log_prob(X_test_t, good=True)
                s = scores.numpy()

        y = y_test.numpy()

        # 6) 指标：
        auc = roc_auc_score(y, s)
        auc_neg = roc_auc_score(y, -s)
        ap = average_precision_score(y, s)

        # 7) 阈值选择（仅报告用；实际应在独立验证集选择阈值，避免测试集信息泄露）
        thr_candidates = np.percentile(s, [i for i in range(1, 100)])
        f1s, accs = [], []
        for th in thr_candidates:
            pred = (s >= th).astype(int)
            f1s.append(f1_score(y, pred))
            accs.append(accuracy_score(y, pred))
        best_idx = int(np.argmax(f1s))
        best_f1, best_acc, best_thr = f1s[best_idx], accs[best_idx], float(thr_candidates[best_idx])

        metrics_all[category] = {
            'AUROC': float(auc),
            'AUROC_neg': float(auc_neg),
            'AP': float(ap),
            'BestF1': float(best_f1),
            'BestAcc': float(best_acc),
            'BestThr': best_thr,
            # 记录关键信息便于复现/对比
            'FeatDim': int(X_train_t.size(1)),
            'ImageSize': int(IMAGE_SIZE),
            'PCA': bool(USE_PCA),
            'ScoreMode': SCORE_MODE,
            'UsePatch': bool(USE_PATCH),
            'PatchSize': int(PATCH_SIZE),
            'PatchStride': int(PATCH_STRIDE),
            'PatchAgg': PATCH_AGG,
            'PatchTopK': int(PATCH_TOPK),
            'UseZipperROI': bool(USE_ZIPPER_ROI),
            'ROIRatio': float(ROI_RATIO),
        }

    print(json.dumps(metrics_all, indent=2))


if __name__ == '__main__':
    # 数据根目录（Image_Anomaly_Detection文件夹）
    data_root = Path(__file__).resolve().parent.parent.parent / 'DM_2025_Dataset' / 'Image_Anomaly_Detection' / 'Image_Anomaly_Detection'
    if not data_root.exists():
        print(f"数据路径不对: {data_root}")
    # 命令行参数，便于快速对比不同设置
    import argparse
    parser = argparse.ArgumentParser(description='ResNet18+Gaussian 异常检测评估')
    parser.add_argument('--categories', type=str, default=','.join(CATEGORIES), help="逗号分隔的类别列表，如 'hazelnut,zipper'")
    parser.add_argument('--image-size', type=int, default=IMAGE_SIZE, help='输入图像尺寸（方形）')
    parser.add_argument('--use-pca', action='store_true', help='启用 PCA 降维')
    parser.add_argument('--no-pca', dest='use_pca', action='store_false', help='禁用 PCA 降维')
    parser.set_defaults(use_pca=USE_PCA)
    parser.add_argument('--pca-dim', type=int, default=PCA_DIM, help='PCA 维度')
    parser.add_argument('--score-mode', type=str, choices=['two-class', 'one-class'], default=SCORE_MODE, help="打分模式：two-class 或 one-class")
    # 滑窗/ROI 参数
    parser.add_argument('--use-patch', action='store_true', help='启用滑窗/patch聚合（更敏感于局部缺陷）')
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE, help='patch 尺寸')
    parser.add_argument('--patch-stride', type=int, default=PATCH_STRIDE, help='patch 步长')
    parser.add_argument('--patch-agg', type=str, choices=['max', 'topk_mean', 'mean'], default=PATCH_AGG, help='patch 聚合方式')
    parser.add_argument('--patch-topk', type=int, default=PATCH_TOPK, help='top-k 聚合的 k')
    parser.add_argument('--use-zipper-roi', action='store_true', help='对 zipper 仅在条带ROI评分')
    parser.add_argument('--roi-ratio', type=float, default=ROI_RATIO, help='ROI 相对宽度比例 (0-1)')

    args = parser.parse_args()

    # 覆盖全局配置
    CATEGORIES = [c.strip() for c in args.categories.split(',') if c.strip()]
    IMAGE_SIZE = args.image_size
    USE_PCA = args.use_pca
    PCA_DIM = args.pca_dim
    SCORE_MODE = args.score_mode
    USE_PATCH = args.use_patch
    PATCH_SIZE = args.patch_size
    PATCH_STRIDE = args.patch_stride
    PATCH_AGG = args.patch_agg
    PATCH_TOPK = args.patch_topk
    USE_ZIPPER_ROI = args.use_zipper_roi
    ROI_RATIO = args.roi_ratio

    train_and_eval(data_root)
