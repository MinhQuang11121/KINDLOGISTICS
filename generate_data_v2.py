import numpy as np

# ==========================================
# CẤU HÌNH THÔNG SỐ LÕI
# ==========================================
num_nodes = 2891     # Ông nhớ check lại số này cho chuẩn với ma trận kề nhé
num_days = 365       # 1 năm
steps_per_day = 96   # 96 khung 15-phút
total_steps = num_days * steps_per_day  # 35.040

print(f"🔥 Bắt đầu sinh khối dữ liệu 6 BIẾN (Có Sự kiện) cho {num_nodes} ngã tư...")

# Khởi tạo khối Tensor 3D (Time, Node, Feature) -> Cấu hình 6 luồng
data_tensor = np.zeros((total_steps, num_nodes, 6), dtype=np.float32)

# ==========================================
# XÂY DỰNG NHU CẦU & SỰ KIỆN (Feature 0 & 5)
# ==========================================
# 1. Nền đơn hàng và Chu kỳ sóng ngày
base_demand = np.random.randint(5, 50, size=(1, num_nodes))
time_idx = np.arange(total_steps)
daily_wave = np.sin(2 * np.pi * time_idx / steps_per_day) + 0.5 * np.sin(2 * np.pi * time_idx / (steps_per_day / 2))
daily_wave = (daily_wave - daily_wave.min()) / (daily_wave.max() - daily_wave.min()) * 1.5 + 0.2
demand = base_demand * daily_wave.reshape(-1, 1)

# 2. Hiệu ứng cuối tuần (Thứ 7, CN x1.5)
for step in range(total_steps):
    if (step // steps_per_day) % 7 >= 5:
        demand[step, :] *= 1.5

# 3. Tạo cờ Sự kiện (Feature 5) & Ép hiệu ứng lên đơn hàng
event_flag = np.zeros((total_steps, num_nodes))
for step in range(total_steps):
    day_of_year = step // steps_per_day
    
    # Ngày Sale đôi (Cứ 30 ngày 1 lần)
    if day_of_year % 30 == 0:
        event_flag[step, :] = 1
        demand[step, :] *= 2.5  # Đơn nổ x2.5
        
    # Lễ lớn (Giả lập mốc 30/4 - 1/5)
    elif 118 <= day_of_year <= 122:
        event_flag[step, :] = 2
        demand[step, :] *= 3.0  # Đơn nổ x3

# 4. Thêm nhiễu ngẫu nhiên và chốt Feature 0, 5
demand += np.random.normal(0, 3, size=(total_steps, num_nodes))
demand = np.clip(demand.astype(int), 0, None)
data_tensor[:, :, 0] = demand
data_tensor[:, :, 5] = event_flag

# ==========================================
# FEATURE 1: NHIỆT ĐỘ (Temperature)
# ==========================================
temp_wave = np.sin(2 * np.pi * (time_idx - 16) / steps_per_day) 
temp = (temp_wave - temp_wave.min()) / (temp_wave.max() - temp_wave.min()) * 12 + 23
data_tensor[:, :, 1] = np.tile(temp.reshape(-1, 1), (1, num_nodes)) + np.random.normal(0, 0.5, size=(total_steps, num_nodes))

# ==========================================
# FEATURE 2: LƯỢNG MƯA (Rainfall)
# ==========================================
rain = np.random.exponential(scale=1.5, size=(total_steps, num_nodes))
rain[np.random.rand(*rain.shape) < 0.90] = 0.0 # 90% tạnh ráo
data_tensor[:, :, 2] = np.clip(rain, 0, None)

# ==========================================
# FEATURE 3: TỐC ĐỘ GIAO THÔNG (Traffic Speed)
# ==========================================
# Tốc độ giảm khi đơn đông (nhất là ngày Lễ/Sale) hoặc trời mưa
base_speed = np.full((total_steps, num_nodes), 40.0)
speed_drop_from_demand = demand * 0.3 
speed_drop_from_rain = rain * 2.0     
speed = base_speed - speed_drop_from_demand - speed_drop_from_rain
data_tensor[:, :, 3] = np.clip(speed, 10, 50) # Kẹt cứng nhất là 10km/h

# ==========================================
# FEATURE 4: THỜI GIAN TRONG NGÀY (Time Index)
# ==========================================
time_feature = (time_idx % steps_per_day) / steps_per_day
data_tensor[:, :, 4] = np.tile(time_feature.reshape(-1, 1), (1, num_nodes))

# ==========================================
# LƯU FILE
# ==========================================
file_name = 'danang_1year_6features.npy'
np.save(file_name, data_tensor)

print("=========================================")
print(f"✅ XONG! Đã tạo file: {file_name}")
print(f"📦 Kích thước (Shape): {data_tensor.shape} -> Chuẩn đét cho in_features = 6")
print("=========================================")