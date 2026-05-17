import osmnx as ox

# 1. Khai báo danh sách các quận trung tâm muốn khoanh vùng
# Bỏ qua huyện Hòa Vang và Hoàng Sa để giảm thiểu số lượng Node nhiễu
quan_trung_tam = [
    'Hải Châu, Da Nang, Vietnam', 
    'Thanh Khê, Da Nang, Vietnam',
    'Sơn Trà, Da Nang, Vietnam', 
    'Cẩm Lệ, Da Nang, Vietnam'
]

print("Đang kéo và khoanh vùng mạng lưới giao thông đô thị Đà Nẵng...")

# 2. OSMnx sẽ tự động khoanh ranh giới (bounding box) của 4 quận này
G_core = ox.graph_from_place(quan_trung_tam, network_type='drive')

# 3. Đếm lại xem sau khi khoanh vùng thì còn bao nhiêu ngã tư
nodes, edges = ox.graph_to_gdfs(G_core)
print(f"Số ngã tư sau khi khoanh vùng: {len(nodes)} (Từ đây gom cụm sẽ ra 2851)")

# 4. VẼ BẢN ĐỒ TRỰC QUAN (Hiện lên một cửa sổ mới)
# Màu nền đen, đường xá màu xanh neon cực ngầu kiểu AI
fig, ax = ox.plot_graph(G_core, node_size=1, node_color='red', edge_color='cyan', bgcolor='black')