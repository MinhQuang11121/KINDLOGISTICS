import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd

print("="*50)
print("🚀 BƯỚC 2: TẠO MA TRẬN KỀ (ADJACENCY MATRIX) CHO 5 QUẬN")
print("="*50)

# 1. Load lại file bản đồ thô vừa tải ban nãy
file_raw = "danang_5quan_raw.graphml"
print(f"Đang đọc cấu trúc đường phố từ {file_raw}...")
G = ox.load_graphml(file_raw)

# 2. Trích xuất danh sách các Ngã tư (Nodes)
nodes, edges = ox.graph_to_gdfs(G)
num_nodes = len(nodes)
print(f"Tổng số Node chốt hạ: {num_nodes}")

# 3. XUẤT FILE TỌA ĐỘ CHO GIAO DIỆN WEB
print("\nĐang xuất file tọa độ cho đội Web...")
toado_df = pd.DataFrame({
    'Node_ID': range(num_nodes), # Đánh số thứ tự từ 0 đến 2890
    'osmid': nodes.index,        # ID gốc của OpenStreetMap
    'y': nodes['y'],             # Vĩ độ (Latitude)
    'x': nodes['x']              # Kinh độ (Longitude)
})
file_csv = 'toado_5quan_ngatu.csv'
toado_df.to_csv(file_csv, index=False)
print(f"✅ Đã lưu: {file_csv} (Nhớ gửi cái này cho Phúc nhé!)")

# 4. ĐÚC MA TRẬN KỀ CHO AI
print("\nĐang đúc Ma trận kề (Adjacency Matrix) để nạp vào STGCN...")
# Khởi tạo một ma trận vuông toàn số 0 (Kích thước 2891 x 2891)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

# Tạo một cuốn từ điển để map ID gốc sang số thứ tự (0 -> 2890)
osm_to_idx = {osmid: i for i, osmid in enumerate(nodes.index)}

# Quét tất cả các con đường (edges) để điền số 1 vào ma trận nếu có đường nối
for u, v, data in G.edges(data=True):
    if u in osm_to_idx and v in osm_to_idx:
        idx_u = osm_to_idx[u]
        idx_v = osm_to_idx[v]
        # Điền số 1 thể hiện 2 ngã tư này thông nhau
        adj_matrix[idx_u, idx_v] = 1.0
        adj_matrix[idx_v, idx_u] = 1.0 # Đường 2 chiều nên đối xứng nhau

# 5. Lưu ma trận ra file .npz
file_npz = 'dist_mat_danang_v2.npz'
np.savez_compressed(file_npz, adj_matrix=adj_matrix) # Tên biến bên trong là adj_matrix
print(f"✅ Đã đúc xong Ma trận kề: {file_npz} (Kích thước: {num_nodes}x{num_nodes})")
print("="*50)
print("🎉 HOÀN TẤT XỬ LÝ DATA! SẴN SÀNG CHO AI HỌC!")