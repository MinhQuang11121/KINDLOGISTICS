import torch
import numpy as np
import matplotlib.pyplot as plt
from model import STGCN

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cấu hình y hệt lúc train
    num_nodes = 2851
    in_features = 5
    seq_len = 12
    pre_len = 3
    
    # 2. Nạp Model và Trọng số đã lưu
    model = STGCN(num_nodes, in_features, 64, seq_len, pre_len).to(device)
    model.load_state_dict(torch.load('models_weights/danang_final_pro.pth'))
    model.eval()
    
    A_danang = torch.eye(num_nodes).to(device)

    # 3. Nạp dữ liệu Test (20% cuối của 1 năm)
    data = np.load('danang_1year_multimodal.npy').astype(np.float32)
    test_start = int(len(data) * 0.8)
    test_data = data[test_start:]

    # 4. Chọn ngẫu nhiên một mốc thời gian và một Node để vẽ
    idx = np.random.randint(0, len(test_data) - 15)
    node_id = np.random.randint(0, num_nodes) # Chọn 1 ngã tư ngẫu nhiên

    # Chuẩn bị dữ liệu đầu vào
    x_raw = test_data[idx : idx + 12, :, :]
    x_tensor = torch.from_numpy(x_raw).float().permute(2, 1, 0).unsqueeze(0).to(device)
    
    # Thực tế (Ground Truth)
    y_true = test_data[idx + 12 : idx + 15, node_id, 0] # Lấy 3 mốc thực tế của kênh đơn hàng

    # Dự báo bằng AI
    with torch.no_grad():
        outputs = model(x_tensor, A_danang) # [1, 2851, 3]
        y_pred = outputs[0, node_id, :].cpu().numpy()

    # 5. VẼ BIỂU ĐỒ SO SÁNH
    plt.figure(figsize=(10, 6))
    time_steps = ['+5 min', '+10 min', '+15 min']
    
    plt.plot(time_steps, y_true, 'g-o', label='Thực tế (Ground Truth)', linewidth=2)
    plt.plot(time_steps, y_pred, 'r--s', label='AI Dự báo (Prediction)', linewidth=2)
    
    plt.title(f'Dự báo nhu cầu tại Ngã tư ID: {node_id}\n(Thời điểm ngẫu nhiên trong tập Test)')
    plt.ylabel('Số lượng đơn hàng / Mật độ')
    plt.xlabel('Tương lai')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Lưu ảnh để dán vào Word
    save_path = f'prediction_sample_node_{node_id}.png'
    plt.savefig(save_path)
    print(f"✅ Đã vẽ xong! Hãy mở file {save_path} để xem kết quả.")
    plt.show()

if __name__ == '__main__':
    visualize_prediction()