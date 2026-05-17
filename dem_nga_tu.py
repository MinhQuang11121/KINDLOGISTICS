import osmnx as ox

# 1. Khai báo khu vực cần kéo dữ liệu
place_name = "Da Nang, Vietnam"
print(f"Đang tải dữ liệu mạng lưới giao thông cho {place_name}...")

# 2. Tải đồ thị (Graph) đường xá dành cho xe cộ (drive)
# Bước này có thể mất khoảng 1-2 phút tùy tốc độ mạng
G = ox.graph_from_place(place_name, network_type='drive')

# 3. Ép kiểu đồ thị sang dạng bảng dữ liệu GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G)

# 4. Đếm tổng số lượng ngã tư (số dòng trong bảng nodes)
so_luong_nga_tu_tho = len(nodes)

print("="*50)
print(f"Tổng số ngã tư (Nodes) thô ban đầu kéo về: {so_luong_nga_tu_tho}")
print("="*50)