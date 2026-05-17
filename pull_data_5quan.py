import osmnx as ox

print("="*50)
print("🚀 KHỞI ĐỘNG HỆ THỐNG KÉO DỮ LIỆU BẢN ĐỒ 5 QUẬN")
print("="*50)

# 1. Khai báo danh sách 5 quận nội thành
quan_trung_tam = [
    'Hải Châu, Da Nang, Vietnam', 
    'Thanh Khê, Da Nang, Vietnam',
    'Sơn Trà, Da Nang, Vietnam', 
    'Cẩm Lệ, Da Nang, Vietnam',
    'Ngũ Hành Sơn, Da Nang, Vietnam'
]

print("\nĐang tải mạng lưới giao thông (Sẽ mất vài phút. Uống ngụm nước đi sếp!)...")

# 2. Tải đồ thị (Graph) từ OpenStreetMap
G_core = ox.graph_from_place(quan_trung_tam, network_type='drive')

# 3. Ép kiểu để đếm số ngã tư thô
nodes, edges = ox.graph_to_gdfs(G_core)
print(f"✅ Đã tải xong! Tổng số ngã tư thô của 5 quận là: {len(nodes)} Node")

# 4. Lưu đồ thị lại thành file cứng để các bước sau (Gom cụm, Ma trận) xài
file_luu = "danang_5quan_raw.graphml"
ox.save_graphml(G_core, file_luu)
print(f"💾 Đã lưu cấu trúc đường phố vào file: {file_luu}")
print("="*50)