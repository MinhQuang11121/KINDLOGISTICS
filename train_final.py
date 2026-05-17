import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH_V_DATA = r'D:\CDIOFN\v_data.npy'
PATH_ADJ = r'D:\CDIOFN\data\cd\dist_mat.npz'
MODEL_SAVE_PATH = r'D:\CDIOFN\stgcn_model_cd.pth'


BATCH_SIZE = 32
NUM_NODES = 6566
EPOCHS = 50
# --- CẤU HÌNH RESUME TRAINING ---
ADDITIONAL_EPOCHS = 15     # Chạy thêm 15 Epoch nữa thôi
NEW_LEARNING_RATE = 0.0001 # Giảm LR xuống 10 lần (từ 1e-3 xuống 1e-4) để ép hội tụ
WEIGHT_DECAY = 1e-4        # Thêm phạt L2 để chống nhiễu
# -------------------------------

print(f"🖥️ Trạng thái: Đang sử dụng {device} (RTX A4000)")

# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH STGCN (GIỮ NGUYÊN)
# ==========================================
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (kernel_size, 1), padding=(1, 0))

    def forward(self, x):
        out = self.conv(x)
        p, q = torch.split(out, out.shape[1] // 2, dim=1)
        return p * torch.sigmoid(q)

class SpatioConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super(SpatioConvLayer, self).__init__()
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, T, N, C]
        out = torch.matmul(self.adj, x) # Graph Conv
        out = torch.matmul(out, self.weight)
        return out.permute(0, 3, 1, 2) # [B, C, T, N]

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj):
        super(STGCNBlock, self).__init__()
        self.tmp1 = TemporalConvLayer(in_channels, hidden_channels)
        self.spa = SpatioConvLayer(hidden_channels, hidden_channels, adj)
        self.tmp2 = TemporalConvLayer(hidden_channels, out_channels)
        self.ln = nn.LayerNorm([NUM_NODES])

    def forward(self, x):
        x = self.tmp1(x)
        x = self.spa(x)
        x = self.tmp2(x)
        return self.ln(x)

class STGCN(nn.Module):
    def __init__(self, adj):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(1, 16, 32, adj)
        self.block2 = STGCNBlock(32, 16, 64, adj)
        self.fcn = nn.Conv2d(64, 3, (1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fcn(x)
        return x[:, :, -1, :] # Lấy mốc thời gian cuối cùng dự báo

# ==========================================
# 3. NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
print("⏳ 1/4 Đang nạp file máu (v_data.npy)...")
v_data = np.load(PATH_V_DATA).astype(np.float32)
mean, std = v_data.mean(), v_data.std()
v_data = (v_data - mean) / (std + 1e-5)

print("⏳ 2/4 Đang nạp file xương (dist_mat.npz)...")
adj_raw = sp.load_npz(PATH_ADJ).toarray()

print("⏳ 3/4 Đang chuẩn hóa ma trận kề...")
adj_id = adj_raw + np.eye(NUM_NODES)
d_inv_sqrt = np.power(np.sum(adj_id, axis=1), -0.5)
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
adj_norm = adj_id * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]
adj_tensor = torch.tensor(adj_norm, dtype=torch.float32).to(device)

print("⏳ 4/4 Đang tạo cửa sổ thời gian (Sliding Window)...")
def sliding_window(data, window_in=12, window_out=3):
    x, y = [], []
    for i in range(len(data) - window_in - window_out):
        x.append(data[i : i + window_in])
        y.append(data[i + window_in : i + window_in + window_out])
    return np.array(x), np.array(y)

x_arr, y_arr = sliding_window(v_data)
X_tensor = torch.tensor(x_arr).unsqueeze(1).to(device)
Y_tensor = torch.tensor(y_arr).to(device)

train_loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. THIẾT LẬP VÀ CHẠY RESUME TRAINING
# ==========================================
model = STGCN(adj_tensor).to(device)

# --- QUAN TRỌNG: LOAD LẠI TRỌNG SỐ TỪ 50 EPOCH TRƯỚC ---
if os.path.exists(MODEL_SAVE_PATH):
    print(f"🔄 Đang nạp lại bộ trọng số cũ từ: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("✅ Đã load thành công! Chuẩn bị ép hội tụ...")
else:
    print("⚠️ Không tìm thấy file trọng số cũ. Sẽ train lại từ đầu!")

# --- Cập nhật Optimizer với Learning Rate cực nhỏ ---
optimizer = torch.optim.Adam(model.parameters(), lr=NEW_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

print(f"🚀 BẮT ĐẦU PHÓNG! Chạy thêm {ADDITIONAL_EPOCHS} Epoch với LR = {NEW_LEARNING_RATE}...")
resume_losses = []

for epoch in range(ADDITIONAL_EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+51}/{EPOCHS + ADDITIONAL_EPOCHS}") # Hiện từ epoch 51
    
    for batch_x, batch_y in progress_bar:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    resume_losses.append(avg_loss)
    print(f"📉 Average Loss Epoch {epoch+51}: {avg_loss:.4f}")

# ==========================================
# 5. LƯU KẾT QUẢ VÀ VẼ ĐỒ THỊ MỚI
# ==========================================
# Ghi đè file model mới (đã xịn hơn)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Vẽ đồ thị 15 Epoch cuối để báo cáo thầy Hùng
plt.figure(figsize=(10, 5))
plt.plot(range(51, 51 + ADDITIONAL_EPOCHS), resume_losses, marker='o', color='b', label='Fine-Tuned Train Loss')
plt.title('Convergence Analysis with Learning Rate Decay (Epoch 51-65)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.legend()
plt.savefig(r'D:\CDIOFN\convergence_loss.png')

print(f"✅ HOÀN HẢO! Đã lưu model và đồ thị hội tụ tại D:\\CDIOFN")