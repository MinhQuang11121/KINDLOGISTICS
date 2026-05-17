import numpy as np
import pandas as pd

# 1. Nạp file dữ liệu
print("Đang mở kho dữ liệu...")
data = np.load('danang_1year_multimodal.npy')

# 2. Trích xuất thử: 10 mốc thời gian đầu tiên của ngày 01/01/2025 tại khu vực trọng điểm (Node số 0)
# Nhắc lại shape: (Thời gian, Node, 5 Kênh)
sample_data = data[0:10, 0, :] 

# 3. Chuyển thành bảng Pandas cho dễ đọc như Excel
df = pd.DataFrame(sample_data, columns=['Lưu lượng (Flow)', 'Mưa (Rain)', 'Nhiệt độ (Temp)', 'Ngày Lễ', 'Ngày Sale'])

print("\n📊 Bảng dữ liệu mẫu (10 mốc thời gian đầu tiên của Ngày 1, Node 0):")
print(df)

# Kiểm tra đỉnh điểm một ngày mưa bão ngẫu nhiên
max_rain_idx = np.argmax(data[:, 0, 1]) # Tìm thời điểm mưa to nhất ở Node 0
print(f"\n🌪️ Lượng mưa lớn nhất ghi nhận được là: {data[max_rain_idx, 0, 1]:.2f} mm")
print(f"🌡️ Nhiệt độ lúc đó: {data[max_rain_idx, 0, 2]:.2f} °C")