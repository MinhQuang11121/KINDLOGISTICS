import pandas as pd
import numpy as np

# Số lượng ngã tư khớp với mô hình STGCN của ông
num_nodes = 2891

# Giới hạn tọa độ khu vực Đà Nẵng (Hải Châu, Thanh Khê, Cẩm Lệ, Sơn Trà, Ngũ Hành Sơn)
lat_min, lat_max = 16.0000, 16.1200
lon_min, lon_max = 108.1500, 108.2500

# Khởi tạo ngẫu nhiên nhưng cố định seed để biểu đồ không bị giật liên tục
np.random.seed(42)
lats = np.random.uniform(lat_min, lat_max, num_nodes)
lons = np.random.uniform(lon_min, lon_max, num_nodes)
node_ids = np.arange(num_nodes)

# Tạo bảng dữ liệu (DataFrame)
df_coords = pd.DataFrame({
    'node_id': node_ids,
    'lat': lats,
    'lon': lons
})

# Lưu thành file CSV
file_name = 'danang_nodes_coords.csv'
df_coords.to_csv(file_name, index=False)

print(f"✅ Đã tạo thành công file '{file_name}'!")
print(f"📍 Tổng số điểm: {len(df_coords)}")
print("Mẫu dữ liệu:")
print(df_coords.head())