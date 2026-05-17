import torch
import numpy as np
from model import STGCN

print("1. Đang nạp Bản đồ và Dữ liệu...")
# Load ma trận bản đồ
A_danang = np.load('danang_adj.npy')
A_tensor = torch.tensor(A_danang, dtype=torch.float32)
num_nodes = A_tensor.shape[0]

# Load dữ liệu để lấy lại mốc chuẩn hóa (mean, std)
traffic_data = np.load('danang_traffic_flow.npy').astype(np.float32)
mean, std = traffic_data.mean(), traffic_data.std()

print("2. Đang đánh thức 'Bộ não' Đà Nẵng...")
model = STGCN(num_nodes=num_nodes)
# Nhớ đảm bảo file danang_finetuned_weights.pth đang nằm cùng thư mục nhé
model.load_state_dict(torch.load('danang_finetuned_weights.pth', map_location='cpu'))
model.eval() # Bật chế độ làm việc (Tắt chế độ học tập)

# ==========================================
# BÀI TEST THỰC TẾ: DỰ ĐOÁN GIỜ CAO ĐIỂM TRƯA
# ==========================================
# Chúng ta lấy 12 mốc (3 tiếng, từ 09:00 đến 12:00) của Ngày đầu tiên
# 09:00 là mốc thứ 36, 12:00 là mốc thứ 48
past_3_hours = traffic_data[36:48] 

# Phải chuẩn hóa (Z-score) thì AI mới hiểu được
past_3_hours_normalized = (past_3_hours - mean) / std

# Ép thành Tensor và thêm chiều Batch = 1 -> Shape: [1, 12, 2851]
X_input = torch.tensor(past_3_hours_normalized).unsqueeze(0)

print("3. Đang suy nghĩ và dự báo tương lai...")
with torch.no_grad():
    prediction_normalized = model(X_input, A_tensor) # Shape: [1, 3, 2851]

# Khôi phục lại con số thực tế (Nhân ngược lại với std và cộng mean)
prediction_real = (prediction_normalized.numpy()[0] * std) + mean
prediction_real = np.clip(np.round(prediction_real), 0, None).astype(int)

# So sánh thực tế và dự đoán tại ngã tư sầm uất nhất (Node có lưu lượng cao nhất)
# Để xem thuật toán PageRank lúc trước chấm Node nào đông nhất
import networkx as nx
G = nx.from_numpy_array(A_danang)
pagerank_scores = nx.pagerank(G, alpha=0.85)
best_node = max(pagerank_scores, key=pagerank_scores.get)

# Lấy 3 mốc thực tế tương ứng (từ 12:00 đến 12:45) để đối chiếu
actual_future = traffic_data[48:51, best_node]

print("\n🎯 KẾT QUẢ DỰ BÁO TẠI NGÃ TƯ SẦM UẤT NHẤT (Node {}):".format(best_node))
print(f"🕒 12:00 - 12:15 | Thực tế: {int(actual_future[0])} chuyến | AI Dự đoán: {prediction_real[0, best_node]} chuyến")
print(f"🕒 12:15 - 12:30 | Thực tế: {int(actual_future[1])} chuyến | AI Dự đoán: {prediction_real[1, best_node]} chuyến")
print(f"🕒 12:30 - 12:45 | Thực tế: {int(actual_future[2])} chuyến | AI Dự đoán: {prediction_real[2, best_node]} chuyến")