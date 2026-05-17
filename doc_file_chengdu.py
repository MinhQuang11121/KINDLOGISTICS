import pickle
import networkx as nx

# 1. Mở file và nạp dữ liệu
file_path = 'data/cd/ChengDu.pkl'

try:
    with open(file_path, 'rb') as f:
        # data_cd phải được gán ở đây
        data_cd = pickle.load(f)
    
    print("--- Thông tin cơ bản về Đồ thị ---")
    
    # 2. Kiểm tra số lượng Nodes (Đây chính là con số 6566 thần thánh đây)
    num_nodes = data_cd.number_of_nodes()
    print(f"Số lượng ngã tư (Nodes): {num_nodes}") 
    
    # 3. Kiểm tra số lượng Edges (Đoạn đường)
    print(f"Số lượng đoạn đường (Edges): {data_cd.number_of_edges()}")

    # 4. Soi thử "cột" (thuộc tính) của 1 Node
    node_id = list(data_cd.nodes)[0]
    print(f"\n--- Dữ liệu tại 1 Node (ID: {node_id}) ---")
    print(f"Các thuộc tính có sẵn: {data_cd.nodes[node_id]}")

    # 5. Soi thử "cột" (thuộc tính) của 1 Edge
    edge_sample = list(data_cd.edges(data=True))[0]
    print(f"\n--- Dữ liệu tại 1 Edge (Đoạn đường) ---")
    print(f"Từ nút {edge_sample[0]} đến {edge_sample[1]}")
    print(f"Các cột dữ liệu: {list(edge_sample[2].keys())}")

except FileNotFoundError:
    print(f"❌ Không tìm thấy file tại: {file_path}. Ông kiểm tra lại đường dẫn nhé!")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")