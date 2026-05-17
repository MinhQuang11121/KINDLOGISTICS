import torch
import numpy as np
from model import STGCN

print("1. Đang tải Ma trận bản đồ Đà Nẵng...")
A_danang = np.load('danang_adj.npy')
A_tensor = torch.tensor(A_danang, dtype=torch.float32)
num_nodes_danang = A_tensor.shape[0] # Chính là 2851

print(f"2. Tạo cơ thể AI mới với {num_nodes_danang} tuyến đường...")
# Khởi tạo mô hình với số nút của Đà Nẵng
danang_model = STGCN(num_nodes=num_nodes_danang)

print("3. Bắt đầu ca phẫu thuật ghép não Thành Đô...")
try:
    # Lắp file weights .pth vừa tải về vào
    danang_model.load_state_dict(torch.load('didi_pretrained_weights.pth', map_location='cpu'))
    print("✅ GHÉP NÃO THÀNH CÔNG! AI đã mang tư duy của Thành Đô vào bản đồ Đà Nẵng!")
    
    # Test thử 1 nhịp thở
    mock_X = torch.randn(16, 12, num_nodes_danang) # Giả lập 1 lô 16 chuỗi dữ liệu
    out = danang_model(mock_X, A_tensor)
    print("👉 Shape dự đoán (Batch, Time_future, Nodes):", out.shape)

except Exception as e:
    print("❌ Ca phẫu thuật thất bại:", e)