import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import osmnx as ox
from sklearn.preprocessing import MinMaxScaler

# Phần 1: Xử lý bộ dữ liệu Pre-train (Dữ liệu DiDi Chuxing)
def preprocess_didi_data(csv_path, grid_size=20, chunksize=100000):
    """
    Xử lý dữ liệu DiDi để tạo feature tensor X (T, N, 1)
    - csv_path: Đường dẫn đến file CSV chứa order_id, start_time, pickup_lat, pickup_lon
    - grid_size: Kích thước lưới (20x20)
    - chunksize: Kích thước chunk để xử lý file lớn (100k rows mỗi chunk)
    """
    # Đọc dữ liệu CSV theo chunks để xử lý file lớn
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk['start_time'] = pd.to_datetime(chunk['start_time'])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Xác định giới hạn bản đồ từ dữ liệu
    min_lat, max_lat = df['pickup_lat'].min(), df['pickup_lat'].max()
    min_lon, max_lon = df['pickup_lon'].min(), df['pickup_lon'].max()

    # Chia lưới 20x20
    lat_bins = np.linspace(min_lat, max_lat, grid_size + 1)
    lon_bins = np.linspace(min_lon, max_lon, grid_size + 1)

    # Gán mỗi điểm vào ô lưới
    df['lat_bin'] = np.digitize(df['pickup_lat'], lat_bins) - 1
    df['lon_bin'] = np.digitize(df['pickup_lon'], lon_bins) - 1
    df['grid_id'] = df['lat_bin'] * grid_size + df['lon_bin']

    # Nhóm theo 15 phút và đếm số đơn hàng trong mỗi ô
    df.set_index('start_time', inplace=True)
    grouped = df.groupby([pd.Grouper(freq='15Min'), 'grid_id']).size().unstack(fill_value=0)

    # Tạo tensor X: (T, N, 1)
    T = len(grouped)
    N = grid_size * grid_size
    X = np.zeros((T, N, 1))
    for t in range(T):
        for n in range(N):
            if n in grouped.columns:
                X[t, n, 0] = grouped.iloc[t, grouped.columns.get_loc(n)] if n < len(grouped.columns) else 0

    return torch.tensor(X, dtype=torch.float32), grouped.index

# Phần 2: Xử lý dữ liệu Fine-tune (Đà Nẵng)
def preprocess_danang_data(order_data, G):
    """
    Xử lý dữ liệu Đà Nẵng: ánh xạ tọa độ vào node gần nhất và tạo adjacency matrix
    - order_data: DataFrame với pickup_lat, pickup_lon
    - G: Đồ thị osmnx
    """
    # Ánh xạ tọa độ vào node gần nhất
    node_ids = []
    for _, row in order_data.iterrows():
        nearest_node = ox.distance.nearest_nodes(G, row['pickup_lon'], row['pickup_lat'])
        node_ids.append(nearest_node)

    order_data['node_id'] = node_ids

    # Tạo adjacency matrix dựa trên khoảng cách
    nodes = list(G.nodes())
    N = len(nodes)
    A = np.zeros((N, N))

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j:
                try:
                    # Tính khoảng cách đường đi ngắn nhất
                    dist = ox.distance.shortest_path_length(G, node1, node2, weight='length')
                    A[i, j] = dist
                except:
                    A[i, j] = np.inf  # Không thể đi đến

    # Chuẩn hóa adjacency matrix (ví dụ: dùng inverse distance)
    A = np.where(A == np.inf, 0, 1 / (A + 1e-6))  # Tránh chia cho 0

    return order_data, torch.tensor(A, dtype=torch.float32), nodes