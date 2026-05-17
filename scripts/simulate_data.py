import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print("1. Đang tải Ma trận không gian Đà Nẵng...")
adj = np.load('danang_adj.npy')
num_nodes = adj.shape[0]  # 2851
num_steps = 960           # 10 ngày * 24 giờ * 4 (15 phút/mốc)

# --- BƯỚC 1: ĐÁNH GIÁ ĐỘ SẦM UẤT (PAGERANK) ---
print("2. Đang dùng Toán học tính toán độ sầm uất của từng ngã tư...")
G = nx.from_numpy_array(adj)
pagerank_scores = nx.pagerank(G, alpha=0.85)

# Chuyển thành mảng và chuẩn hóa (0 đến 1)
spatial_weights = np.array([pagerank_scores[i] for i in range(num_nodes)])
spatial_weights = spatial_weights / spatial_weights.max()

# --- BƯỚC 2: TẠO SÓNG THỜI GIAN (GAUSSIAN WAVES) ---
print("3. Đang giả lập nhịp đập giao nhận trong 10 ngày...")
traffic = np.zeros((num_steps, num_nodes))
time_steps = np.arange(num_steps)

for day in range(10):
    day_offset = day * 96  # 1 ngày có 96 mốc 15 phút
    
    # Sóng 1: Sáng đi làm (Mốc 32 ~ 8h00 sáng)
    m_peak = np.exp(-0.5 * ((time_steps - (day_offset + 32)) / 4)**2) * 1.0
    
    # Sóng 2: Bùng nổ đơn đồ ăn trưa (Mốc 48 ~ 12h00 trưa) - Sóng cao nhất
    n_peak = np.exp(-0.5 * ((time_steps - (day_offset + 48)) / 6)**2) * 1.5
    
    # Sóng 3: Chiều tan tầm (Mốc 72 ~ 18h00 chiều)
    e_peak = np.exp(-0.5 * ((time_steps - (day_offset + 72)) / 8)**2) * 1.3
    
    daily_wave = m_peak + n_peak + e_peak
    
    # --- BƯỚC 3: TRỘN KHÔNG GIAN, THỜI GIAN VÀ NHIỄU ---
    for i in range(num_nodes):
        # Nền tảng (base) + Sóng (wave) * Độ sầm uất + Nhiễu ngẫu nhiên
        noise = np.random.normal(0, 0.2, num_steps) # Nhiễu 20%
        # Giả định đường đông nhất có max ~ 40 lượt/15 phút
        node_traffic = (daily_wave * spatial_weights[i] * 40) + np.abs(noise) + (spatial_weights[i] * 5)
        traffic[:, i] += node_traffic

# Làm tròn thành số nguyên (không có nửa chiếc xe/đơn hàng)
traffic = np.clip(traffic, 0, None).astype(np.int32)

# Lưu lại mỏ vàng này
np.save('danang_traffic_flow.npy', traffic)
print(f"✅ HOÀN THÀNH! Đã lưu ma trận lưu lượng Đà Nẵng: danang_traffic_flow.npy | Shape: {traffic.shape}")

# =====================================================================
# VẼ BIỂU ĐỒ CHUẨN MỰC BÁO CÁO (TRỤC X THEO GIỜ 24H)
# =====================================================================
best_node = np.argmax(spatial_weights)

plt.figure(figsize=(10, 5)) # Mở rộng khung hình cho thoáng
plt.plot(traffic[:96, best_node], color='#ff7f0e', linewidth=2.5, label='Lượng giao nhận')
plt.title(f"Mô phỏng nhịp độ giao nhận 1 ngày tại nút giao sầm uất nhất", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Thời gian trong ngày", fontsize=12)
plt.ylabel("Số lượng đơn hàng / Chuyến xe", fontsize=12)

# Thiết lập trục X theo chuẩn 24h
# 1 ngày có 96 mốc (15 phút/mốc). Cứ 4 mốc = 1 tiếng.
tick_positions = [0, 16, 32, 48, 64, 80, 95]
tick_labels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:45']
plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)

# Thêm lưới cho dễ nhìn
plt.grid(True, linestyle='--', alpha=0.6)

# Đánh dấu các đỉnh sóng để làm nổi bật
plt.axvline(x=48, color='red', linestyle=':', alpha=0.5) # Điểm 12h trưa
plt.axvline(x=72, color='red', linestyle=':', alpha=0.5) # Điểm 18h chiều

plt.legend()
plt.tight_layout() # Căn lề tự động để không bị cắt chữ
plt.savefig('simulated_wave_24h.png', dpi=300) # Lưu ảnh chất lượng cao 300 DPI
print("👉 Đã xuất biểu đồ chuẩn 24h siêu nét ra file 'simulated_wave_24h.png'!")