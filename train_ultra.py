import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt 
from model import STGCN 

def train_ultra():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" ULTRA TRAINING - ĐÀ NẴNG (TRANSFER LEARNING + GRADIENT CLIPPING)")
    
    num_nodes = 2891       
    in_features = 6        
    hidden_features = 128  
    num_layers = 3         
    batch_size = 8 
    epochs = 100

    print("Đang nạp khối dữ liệu 6 biến...")
    data = np.load('danang_1year_6features.npy').astype(np.float32)
    
    # --- CHIA TẬP DATA 80/20 (Train / Validation) ---
    train_size = int(len(data) * 0.8)
    
    # --- FIX LỖI Z-SCORE: CHUẨN HÓA THEO TỪNG ĐẶC TRƯNG ---
    print("Đang chuẩn hóa dữ liệu Z-Score theo từng Feature...")
    # Thêm axis=(0, 1) để tính mean/std cho riêng 6 features
    mean = np.mean(data[:train_size], axis=(0, 1), keepdims=True)
    std = np.std(data[:train_size], axis=(0, 1), keepdims=True)
    
    # Lưu lại mean và std của biến Traffic (index 0) để sau này Inverse (Giải chuẩn hóa) khi tính MAE, RMSE
    np.save('traffic_mean_std.npy', np.array([mean[0, 0, 0], std[0, 0, 0]]))
    
    data = (data - mean) / (std + 1e-5)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Đã chia dữ liệu: Train ({len(train_data)} mẫu), Val ({len(val_data)} mẫu)")
    
    model = STGCN(num_nodes, in_features, hidden_features, 12, 3, num_layers).to(device)
    
    print("Đang nạp bản đồ không gian 5 quận...")
    adj_data = np.load('dist_mat_danang_v2.npz')
    A_danang = torch.from_numpy(adj_data['adj_matrix']).float().to(device)

    # --- TRANSFER LEARNING ---
    PRETRAINED_PATH = r'D:\CDIOFN\stgcn_model_cd.pth' 

    if os.path.exists(PRETRAINED_PATH):
        print(f"🔄 Đang chuyển giao tri thức từ file: {PRETRAINED_PATH} ...")
        pretrained_dict = torch.load(PRETRAINED_PATH, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "ln" not in k and "fcn" not in k}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print("✅ Transfer Learning thành công!")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file trọng số!")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    history_train_loss = []
    history_val_loss = []

    print(f"\n☢️ BẮT ĐẦU HUẤN LUYỆN!\n")
    
    for epoch in range(epochs):
        # ==========================================
        # PHASE 1: TRAINING
        # ==========================================
        model.train()
        start_time = time.time()
        epoch_train_loss = []
        
        for step in range(150): 
            indices = np.random.randint(0, train_data.shape[0] - 15, batch_size)
            x_batch, y_batch = [], []
            for idx in indices:
                x_batch.append(train_data[idx : idx + 12, :, :])
                y_batch.append(train_data[idx + 12 : idx + 15, :, 0]) # Target là biến 0 (Traffic)
            
            x_t = torch.from_numpy(np.array(x_batch)).permute(0, 3, 2, 1).to(device)
            y_t = torch.from_numpy(np.array(y_batch)).permute(0, 2, 1).to(device)

            optimizer.zero_grad()
            out = model(x_t, A_danang)
            
            # Đảm bảo output và target cùng kích thước trước khi tính loss
            if out.shape != y_t.shape:
                out = out.view_as(y_t)
                
            loss = criterion(out, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_train_loss.append(loss.item())

        avg_train_loss = np.mean(epoch_train_loss)
        history_train_loss.append(avg_train_loss)

        # ==========================================
        # PHASE 2: VALIDATION (FIX TỐC ĐỘ)
        # ==========================================
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            # Xử lý Validation theo Batch thay vì từng mẫu một
            val_indices = np.random.randint(0, val_data.shape[0] - 15, 50)
            x_val_batch, y_val_batch = [], []
            for idx in val_indices:
                x_val_batch.append(val_data[idx : idx + 12, :, :])
                y_val_batch.append(val_data[idx + 12 : idx + 15, :, 0])
                
            x_t_val = torch.from_numpy(np.array(x_val_batch)).permute(0, 3, 2, 1).to(device)
            y_t_val = torch.from_numpy(np.array(y_val_batch)).permute(0, 2, 1).to(device)
            
            out_val = model(x_t_val, A_danang)
            if out_val.shape != y_t_val.shape:
                out_val = out_val.view_as(y_t_val)
                
            v_loss = criterion(out_val, y_t_val)
            val_loss_list.append(v_loss.item())
        
        avg_val_loss = np.mean(val_loss_list)
        history_val_loss.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"🚀 Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | Time: {time.time()-start_time:.1f}s")

    # ==========================================
    # KẾT THÚC HUẤN LUYỆN: LƯU MODEL VÀ VẼ ĐỒ THỊ
    # ==========================================
    os.makedirs('models_weights', exist_ok=True)
    save_path = 'models_weights/danang_ultra_v6.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n👑 HUẤN LUYỆN HOÀN TẤT! Đã lưu trọng số tại: {save_path}")

    # XUẤT BIỂU ĐỒ
    print("📈 Đang xuất biểu đồ hội tụ...")
    plt.figure(figsize=(10, 6))
    plt.plot(history_train_loss, label='Train Loss (MSE)', color='blue', linewidth=2)
    plt.plot(history_val_loss, label='Validation Loss (MSE)', color='red', linestyle='--', linewidth=2)
    plt.title('Đồ thị Hội tụ STGCN - Không gian Đà Nẵng (Đa Đặc Trưng)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('danang_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu ảnh đồ thị tại: danang_loss_curve.png")

# PHẦN GỌI HÀM NẰM Ở ĐÂY ĐỂ TRÁNH LỖI KHÔNG CHẠY
if __name__ == '__main__':
    train_ultra()