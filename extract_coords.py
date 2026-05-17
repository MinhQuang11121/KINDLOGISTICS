import osmnx as ox
import pandas as pd

# 1. Định nghĩa khu vực (Khớp với khu vực ông làm ma trận kề)
place_name = "Da Nang, Vietnam"
districts = ["Hai Chau District", "Thanh Khe District", "Sơn Trà District", "Ngũ Hành Sơn District", "Cẩm Lệ District"]
places = [f"{d}, Da Nang, Vietnam" for d in districts]

print("🌐 Đang tải mạng lưới giao thông từ OpenStreetMap...")
# Lấy mạng lưới đường bộ cho xe máy/ô tô (drive)
graph = ox.graph_from_place(places, network_type='drive')

# 2. Trích xuất các Node (Ngã tư)
nodes, edges = ox.graph_to_gdfs(graph)

# 3. Chỉ lấy các cột cần thiết: ID ngã tư, Vĩ độ (Lat), Kinh độ (Lon)
nodes_data = nodes[['y', 'x']].reset_index()
nodes_data.columns = ['node_id', 'lat', 'lon']

# --- LƯU Ý QUAN TRỌNG ---
# Vì mô hình của ông đang chạy cố định 2891 node, 
# ta cần đảm bảo số lượng node trích xuất ra phải khớp chính xác.
# Nếu số lượng khác, ông hãy lấy đúng 2891 node đầu tiên hoặc lọc theo logic cũ.
nodes_data = nodes_data.head(2891) 

# 4. Xuất file CSV
nodes_data.to_csv('danang_nodes_coords.csv', index=False)

print(f"✅ Đã tạo xong file: danang_nodes_coords.csv")
print(f"📍 Tổng số điểm tọa độ: {len(nodes_data)}")
print(nodes_data.head())