import os
import torch
import numpy as np
from model import STGCN

def test_ultra():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔍 ĐANG KHỞI ĐỘNG CHƯƠNG TRÌNH KIỂM THỬ (TESTING)...")

    # 1. Cấu hình kiến trúc giống hệt lúc Train
    num_nodes = 2891
    in_features = 6
    hidden_features = 128
    num_layers = 3

    # 2. Nạp dữ liệu
    print("Đang nạp dữ liệu và ma trận kề...")
    data = np.load('danang_1year_6features.npy').astype(np.float32)
    train_size = int(len(data) * 0.8)

    # Lấy Mean/Std TỪ TẬP TRAIN để chuẩn hóa tập Test (Nguyên tắc vàng trong ML)
    mean = np.mean(data[:train_size], axis=(0, 1), keepdims=True)
    std = np.std(data[:train_size], axis=(0, 1), keepdims=True)
    
    # Lưu riêng Mean và Std của biến số 0 (Traffic / Lưu lượng) để lát nữa giải chuẩn hóa
    traffic_mean = mean[0, 0, 0]
    traffic_std = std[0, 0, 0]

    # Chuẩn hóa toàn bộ
    data_norm = (data - mean) / (std + 1e-5)
    test_data = data_norm[train_size:] # Đây là tập dữ liệu AI chưa từng thấy

    # 3. Khởi tạo Model và nạp Trọng số (Weights)
    model = STGCN(num_nodes, in_features, hidden_features, 12, 3, num_layers).to(device)
    model_path = 'models_weights/danang_ultra_v6.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Đã nạp thành công bộ não AI từ: {model_path}")
    else:
        print(f"❌ Lỗi: Không tìm thấy file {model_path}! Hãy chắc chắn ông đã chạy xong train_ultra.py")
        return

    model.eval()

    adj_data = np.load('dist_mat_danang_v2.npz')
    A_danang = torch.from_numpy(adj_data['adj_matrix']).float().to(device)

    # 4. Quét qua tập Test để dự báo
    print("🚀 Đang chạy dự báo trên tập Validation/Test...")
    predictions = []
    ground_truths = []

    with torch.no_grad():
        # Quét qua dữ liệu test với bước nhảy batch
        test_indices = range(0, len(test_data) - 15, 32) 
        for idx in test_indices:
            x_test = test_data[idx : idx + 12, :, :]
            y_test = test_data[idx + 12 : idx + 15, :, 0] # Ground truth (Traffic)

            # Thêm chiều batch_size = 1 và chuyển vị
            x_t = torch.from_numpy(x_test).unsqueeze(0).permute(0, 3, 2, 1).to(device)

            out = model(x_t, A_danang)

            # Đưa kết quả về CPU
            pred_val = out.cpu().numpy().squeeze() 
            truth_val = y_test.T

            # GIẢI CHUẨN HÓA: Phục hồi số lượng đơn hàng thực tế
            pred_real = (pred_val * traffic_std) + traffic_mean
            truth_real = (truth_val * traffic_std) + traffic_mean

            predictions.append(pred_real)
            ground_truths.append(truth_real)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # 5. TÍNH TOÁN CÁC CHỈ SỐ
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths)**2))
    
    # MAPE cần cẩn thận lỗi chia cho 0 nếu có ngã tư không có đơn hàng nào
    epsilon = 1e-5
    mape = np.mean(np.abs((ground_truths - predictions) / (ground_truths + epsilon))) * 100

    print("\n" + "="*50)
    print("📊 BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ĐƯA VÀO CHƯƠNG 4)")
    print("="*50)
    print(f"🔹 MAE  (Sai số tuyệt đối trung bình) : {mae:.4f} (đơn hàng/lượt)")
    print(f"🔹 RMSE (Căn bậc hai sai số)          : {rmse:.4f} (đơn hàng/lượt)")
    print(f"🔹 MAPE (Sai số phần trăm)            : {mape:.2f} %")
    print("="*50)
    print("💡 Ghi chú: Các chỉ số đã được giải chuẩn hóa về đơn vị đo gốc.")

if __name__ == '__main__':
    test_ultra()