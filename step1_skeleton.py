import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

# Cấu hình đường dẫn
base_path = r'D:\CDIOFN\data\cd'
path_txt = os.path.join(base_path, 'road_distance_direction.txt')
path_npz = os.path.join(base_path, 'dist_mat.npz')

print("⏳ Bước 1: Đang xay file 1.1GB... (Mất tầm 3-5 phút)")

# 1. Đọc dữ liệu (From, To, Distance)
df = pd.read_csv(path_txt, header=None)

# 2. Chuyển khoảng cách thành trọng số (Gaussian Kernel)
distances = df[2].replace([np.inf, -np.inf], 999999).values
sigma = np.std(distances[distances < 999999])
weights = np.exp(-(distances**2) / (sigma**2))

# 3. Lọc bỏ các kết nối không hợp lệ
mask = (df[2] != np.inf) & (weights > 0.01)
row = df[0].values[mask].astype(int)
col = df[1].values[mask].astype(int)
data = weights[mask]

# 4. Lưu ma trận thưa 6566 x 6566
adj = sp.coo_matrix((data, (row, col)), shape=(6566, 6566))
sp.save_npz(path_npz, adj.tocsr())

print(f"✅ Xong Bước 1! Đã có file: {path_npz}")