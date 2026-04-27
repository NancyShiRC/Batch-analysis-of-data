import os
import yaml
import pandas as pd
import numpy as np
import openslide
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

wsi_dir = cfg["data"]["wsi_dir"]
att_dir = cfg["data"]["att_dir"]
out_dir = cfg["output"]["out_dir"]

patch_size = cfg["params"]["patch_size"]
level = cfg["params"]["level"]
top_ratio = cfg["params"]["top_ratio"]

k = cfg["spatial"]["knn_k"]
q = cfg["spatial"]["eps_quantile"]
min_samples = cfg["spatial"]["min_samples"]
fallback_q = cfg["spatial"]["fallback_quantile"]

os.makedirs(out_dir, exist_ok=True)

# =========================
# Ki-67 intensity
# =========================
def ki67_intensity(patch):
    patch = patch.astype(np.float32) + 1
    od = -np.log(patch / 255.0)
    return np.mean(od[:, :, 0] + od[:, :, 1]) / 2

# =========================
# 找WSI
# =========================
def find_wsi(sample_id):
    for f in os.listdir(wsi_dir):
        if sample_id in f and f.endswith(".svs"):
            return os.path.join(wsi_dir, f)
    return None

# =========================
# 自适应 eps
# =========================
def estimate_eps(coords, k=5, q=0.9):
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    kth_dist = distances[:, -1]
    return np.quantile(kth_dist, q)

# =========================
# Moran’s I
# =========================
def moran_like(x, coords):
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    w = 1 / dist
    w[np.isinf(w)] = 0

    x_mean = x.mean()
    num = 0
    den = np.sum((x - x_mean) ** 2)

    for i in range(len(x)):
        for j in range(len(x)):
            num += w[i, j] * (x[i] - x_mean) * (x[j] - x_mean)

    return (len(x) / np.sum(w)) * (num / den)

# =========================
# 主循环
# =========================
results = []

files = [f for f in os.listdir(att_dir) if f.endswith("_attention.csv")]

for file in tqdm(files):

    sample_id = file.replace("_attention.csv", "")
    att_path = os.path.join(att_dir, file)
    wsi_path = find_wsi(sample_id)

    if wsi_path is None:
        continue

    slide = openslide.OpenSlide(wsi_path)
    df = pd.read_csv(att_path)

    intensities, attentions, coords = [], [], []

    # ===== 单次读取patch =====
    for _, row in df.iterrows():
        x = int(row["x_coordinate"])
        y = int(row["y_coordinate"])
        att = float(row["attention_score"])

        try:
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patch = np.array(patch)[:, :, :3]
        except:
            continue

        intensities.append(ki67_intensity(patch))
        attentions.append(att)
        coords.append([x, y])

    if len(intensities) < 10:
        continue

    intensities = np.array(intensities)
    attentions = np.array(attentions)
    coords = np.array(coords)

    # =========================
    # Part 1：attention-based
    # =========================
    att_norm = attentions / (np.sum(attentions) + 1e-8)

    k_top = int(len(attentions) * top_ratio)
    idx = np.argsort(attentions)

    high_idx = idx[-k_top:]
    low_idx = idx[:k_top]

    high_mean = np.mean(intensities[high_idx])
    low_mean = np.mean(intensities[low_idx])

    delta_att = high_mean - low_mean
    weighted_ki67 = np.sum(att_norm * intensities)
    corr, _ = spearmanr(attentions, intensities)

    # =========================
    # Part 2：DBSCAN spatial
    # =========================
    eps = estimate_eps(coords, k=k, q=q)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # fallback机制
    if len(set(labels)) <= 1:
        threshold = np.quantile(attentions, fallback_q)
        hotspot_mask = attentions >= threshold
    else:
        df_tmp = pd.DataFrame({"att": attentions, "label": labels})
        cluster_mean = df_tmp.groupby("label")["att"].mean()
        hot_cluster = cluster_mean.idxmax()
        hotspot_mask = labels == hot_cluster

    if np.sum(hotspot_mask) == 0 or np.sum(~hotspot_mask) == 0:
        delta_spatial = np.nan
        rho = np.nan
    else:
        hotspot_ki67 = np.mean(intensities[hotspot_mask])
        non_hotspot_ki67 = np.mean(intensities[~hotspot_mask])
        delta_spatial = hotspot_ki67 - non_hotspot_ki67

        centroid = coords[hotspot_mask].mean(axis=0).reshape(1, -1)
        dist = cdist(coords, centroid).flatten()
        rho, _ = spearmanr(dist, intensities)

    I_ki67 = moran_like(intensities, coords)
    I_att = moran_like(attentions, coords)

    # =========================
    # 保存
    # =========================
    results.append({
        "case_id": sample_id,
        "delta_att_ki67": delta_att,
        "weighted_ki67": weighted_ki67,
        "corr_att_ki67": corr,
        "delta_spatial": delta_spatial,
        "distance_rho": rho,
        "moran_ki67": I_ki67,
        "moran_att": I_att
    })

# =========================
# 输出
# =========================
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(out_dir, "final_results.csv"), index=False)

print("\n===== SUMMARY =====")
print(df_out.describe())
print("\nSaved to:", out_dir)